"""
Tactical Radar & Multi-Sensor Fusion GUI Package.
"""

def launch_gui() -> None:
    """Convenience launcher for the tactical radar console application."""
    from .app import main
    main()


__all__ = ["launch_gui"]
