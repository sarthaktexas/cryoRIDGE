from chimerax.core.tools import ToolInstance


class MapReliabilityTool(ToolInstance):
    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from .ui import MapReliabilityWindow

        self.tool_window = MapReliabilityWindow(self)
