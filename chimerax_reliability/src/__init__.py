# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MapReliabilityAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from chimerax.core.commands import register
        from . import cmd

        desc = cmd.reliability_desc
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, cmd.reliability)

    @staticmethod
    def start_tool(session, bundle_info, tool_info):
        from chimerax.core import tools
        from .tool import MapReliabilityTool

        return tools.get_singleton(session, MapReliabilityTool, tool_info.name, create=True)


bundle_api = _MapReliabilityAPI()
