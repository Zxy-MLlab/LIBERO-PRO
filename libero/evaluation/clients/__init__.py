"""Built-in clients and their registration side effects."""

from .registry import available_clients, create_client, register_client
from .gr00t_client import Gr00tClient
from .openpi_client import OpenPIClient
from .openvla_client import OpenVLAClient

__all__ = [
    "Gr00tClient",
    "OpenPIClient",
    "OpenVLAClient",
    "available_clients",
    "create_client",
    "register_client",
]
