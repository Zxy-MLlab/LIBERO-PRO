#!/usr/bin/env python3
"""Compose LIBERO agent, wrist, and episode metadata into review videos."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (768, 512)
PANEL_SIZE = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path, help="Directory containing episodes.jsonl and videos/")
    return parser.parse_args()


def find_font(bold=False):
    names = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return name
    command = ["fc-match", "-f", "%{file}", "DejaVu Sans:style={}".format("Bold" if bold else "Book")]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    path = result.stdout.strip()
    if not path:
        raise RuntimeError("could not locate a font for the information panel")
    return path


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSON on {} line {}".format(path, line_number)) from exc
    return records


def load_config(evaluation_dir):
    path = evaluation_dir / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}


def resolve_video_paths(evaluation_dir, record):
    agent = Path(record.get("video_path", ""))
    if not agent.is_absolute():
        agent = evaluation_dir / agent
    wrist_value = record.get("wrist_video_path", "")
    wrist = Path(wrist_value) if wrist_value else agent.with_name(agent.stem + "_wrist.mp4")
    if not wrist.is_absolute():
        wrist = evaluation_dir / wrist
    return agent, wrist


def fit_text(draw, text, font, max_width, max_lines):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    consumed = " ".join(lines)
    if len(consumed) < len(" ".join(words)) and lines:
        while lines[-1] and draw.textbbox((0, 0), lines[-1] + "…", font=font)[2] > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def render_panel(path, record, prompt, regular_font, bold_font):
    image = Image.new("RGB", PANEL_SIZE, "#10151d")
    draw = ImageDraw.Draw(image)
    label = ImageFont.truetype(regular_font, 12)
    body = ImageFont.truetype(regular_font, 15)
    strong = ImageFont.truetype(bold_font, 16)
    small = ImageFont.truetype(regular_font, 13)

    draw.rectangle((0, 0, 255, 3), fill="#55b8ff")
    # Leave a small breathing space below the wrist frame.
    draw.line((16, 44, 240, 44), fill="#2a3544", width=1)

    suite = str(record.get("suite", "unknown"))
    task_id = record.get("task_id", "?")
    draw.text((16, 56), "{}  ·  TASK {}".format(suite, task_id), font=strong, fill="#eef4fb")

    episode_id = str(record.get("episode_id", "?"))
    episode_number = episode_id.rsplit("/", 1)[-1]
    if episode_number.isdigit():
        episode_number = str(int(episode_number) + 1)
    init_state = record.get("init_state_id", "?")
    draw.text((16, 82), "Episode {}  ·  Init {}".format(episode_number, init_state), font=small, fill="#aab8c8")

    draw.text((16, 112), "PROMPT", font=label, fill="#7f91a8")
    for index, line in enumerate(fit_text(draw, prompt, body, 224, 4)):
        draw.text((16, 132 + index * 20), line, font=body, fill="#eef4fb")

    success = bool(record.get("success", False))
    status = "SUCCESS" if success else "FAILURE"
    color = "#4ade80" if success else "#fb7185"
    draw.rounded_rectangle((158, 13, 240, 37), radius=5, fill="#19222d", outline=color, width=1)
    status_x = 169 if success else 168
    draw.text((status_x, 17), status, font=label, fill=color)
    steps = record.get("steps", "?")
    draw.text((16, 232), "{} total steps".format(steps), font=label, fill="#7f91a8")
    image.save(path)


def ffmpeg_filter(stride, final_step, font_path):
    step_expression = "%" + ("{eif\\:min(n*%d+1\\,%d)\\:d}" % (stride, final_step))
    return (
        "[0:v]scale=512:512:force_original_aspect_ratio=decrease,"
        "pad=512:512:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[agent];"
        "[1:v]scale=256:256:force_original_aspect_ratio=decrease,"
        "pad=256:256:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[wrist];"
        "[2:v]scale=256:256,setsar=1[panel];"
        "[wrist][panel]vstack=inputs=2:shortest=1[right];"
        "[agent][right]hstack=inputs=2:shortest=1[layout];"
        "[layout]drawtext=fontfile='{}':text='STEP {}':x=528:y=269:"
        "fontsize=23:fontcolor=white[v]"
    ).format(font_path.replace("'", "\\'"), step_expression)


def compose_episode(record, evaluation_dir, config, fonts, temp_dir):
    agent, wrist = resolve_video_paths(evaluation_dir, record)
    if not agent.is_file() or not wrist.is_file():
        raise FileNotFoundError("missing paired videos: {} / {}".format(agent, wrist))
    output = temp_dir / (agent.stem + "_overview.mp4")

    prompt = record.get("prompt") or record.get("task", "").replace("_", " ")
    panel = temp_dir / (agent.stem + "_panel.png")
    render_panel(panel, record, prompt, fonts[0], fonts[1])
    stride = max(1, int(config.get("recording", {}).get("stride", 1)))
    fps = max(1, int(config.get("recording", {}).get("fps", 10)))
    final_step = max(1, int(record.get("steps", 1)))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(agent), "-i", str(wrist), "-loop", "1", "-framerate", str(fps), "-i", str(panel),
        "-filter_complex", ffmpeg_filter(stride, final_step, fonts[1]),
        "-map", "[v]", "-shortest", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)
    return output


def episode_index(record):
    value = str(record.get("episode_id", "")).rsplit("/", 1)[-1]
    return int(value) if value.isdigit() else 0


def concat_task_videos(task_id, episodes, output_dir, temp_dir):
    ordered = sorted(episodes, key=lambda item: episode_index(item[0]))
    list_path = temp_dir / "task_{}_episodes.txt".format(task_id)
    lines = []
    for _, video_path in ordered:
        escaped = str(video_path.resolve()).replace("'", "'\\''")
        lines.append("file '{}'".format(escaped))
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = output_dir / "task_{}.mp4".format(task_id)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy", "-movflags", "+faststart", str(output),
    ], check=True)
    return output


def main():
    args = parse_args()
    evaluation_dir = args.evaluation_dir.resolve()
    episodes_path = evaluation_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError("missing {}".format(episodes_path))
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    output_dir = evaluation_dir / "render"
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_overview.mp4", "task_*_episodes.mp4"):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()
    records = load_records(episodes_path)
    config = load_config(evaluation_dir)
    fonts = (find_font(False), find_font(True))
    created = failed = 0
    task_videos = {}
    with tempfile.TemporaryDirectory(prefix="libero-eval-video-") as temp_name:
        temp_dir = Path(temp_name)
        for record in records:
            try:
                output = compose_episode(record, evaluation_dir, config, fonts, temp_dir)
                created += 1
                task_videos.setdefault(record.get("task_id", "unknown"), []).append((record, output))
            except Exception as exc:
                failed += 1
                print("failed: {} ({})".format(record.get("episode_id", "unknown"), exc))
        for task_id, episodes in sorted(task_videos.items(), key=lambda item: str(item[0])):
            try:
                output = concat_task_videos(task_id, episodes, output_dir, temp_dir)
                print("concatenated: {}".format(output))
            except Exception as exc:
                failed += 1
                print("failed to concatenate task {}: {}".format(task_id, exc))
    print("complete: tasks={} failed={}".format(len(task_videos), failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
