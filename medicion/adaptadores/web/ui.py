"""Interfaz web (HTML + CSS + JavaScript, sin dependencias externas). El texto vive en ui.html."""
import pkgutil

HTML = pkgutil.get_data(__package__, "ui.html").decode("utf-8")
