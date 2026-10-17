project = "scorpus"
copyright = "2026, Li Lab"
author = "Li Lab"

extensions = [
    "myst_nb",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# Walkthrough notebooks need the demo data and corpora that aren't available
# during a doc build, so notebooks render their already stored outputs rather
# than being re-executed here.
nb_execution_mode = "off"

html_theme = "furo"
html_title = "scorpus"
html_static_path = ["assets"]
html_favicon = "assets/scorpus_favicon.png"
html_theme_options = {
    "light_logo": "scorpus_logo_light.png",
    "dark_logo": "scorpus_logo.png",
    "sidebar_hide_name": True,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
