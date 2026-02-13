import imas  # noqa: F401 -- module required in doctests
from imas.ids_primitive import IDSPrimitive
from imas.ids_struct_array import IDSStructArray
from imas.util import get_parent


def get_dynamic_aos_ancestor(ids_node: IDSPrimitive) -> IDSStructArray:
    """Returns the dynamic Arrays of Structures ancestor for the provided node.

    Examples:
        >>> cp = imas.IDSFactory("4.0.0").core_profiles()
        >>> cp.profiles_1d.resize(1)
        >>> get_dynamic_aos_ancestor(cp.profiles_1d[0].zeff) is cp.profiles_1d
        True
        >>> eq = imas.IDSFactory("4.0.0").equilibrium()
        >>> eq.time_slice.resize(1)
        >>> eq.time_slice[0].profiles_2d.resize(1)
        >>> aos_ancestor = get_dynamic_aos_ancestor(eq.time_slice[0].profiles_2d[0].psi)
        >>> aos_ancestor is eq.time_slice
        True
    """
    node = get_parent(ids_node)
    while node is not None and (
        not isinstance(node, IDSStructArray)
        or not node.metadata.coordinates[0].is_time_coordinate
    ):
        node = get_parent(node)
    if node is None:
        raise RuntimeError(
            f"IDS node {ids_node} is not part of a time-dependent Array of Structures."
        )
    return node


def get_path_from_aos(path: str, aos: IDSStructArray) -> str:
    """Get the component of path relative to the provided Arrays of Structures ancestor.

    Examples:
        >>> cp = imas.IDSFactory("4.0.0").core_profiles()
        >>> get_path_from_aos("profiles_1d[0]/ion[1]/temperature", cp.profiles_1d)
        'ion[1]/temperature'
    """
    path_parts = path.split("/")
    aos_parts = aos.metadata.path.parts
    return "/".join(path_parts[len(aos_parts) :])
