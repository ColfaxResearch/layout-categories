"""Shared scene base class for layout-categories animations."""

from manim import FadeOut, Scene

from .style import BACKGROUND


class LayoutScene(Scene):
    """Scene with the layout-categories house style applied."""

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BACKGROUND

    def clear_scene(self, *, last: bool = False) -> None:
        """Fade everything out between examples, pausing unless it is the last."""
        self.play(*(FadeOut(m) for m in self.mobjects), run_time=0.6)
        self.clear()
        if not last:
            self.wait(0.2)
