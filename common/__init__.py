"""Code shared by the controller and the runners.

Deliberately dependency-free: the controller installs without torch and must
stay that way, so nothing in here may import a training library.
"""
