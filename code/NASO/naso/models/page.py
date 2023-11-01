class PageSetup:
    title = ""
    description = ""
    actions: list[tuple[str, str, str]] = []

    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self.actions = []

    def add_pageaction(self, url: str, title: str, color: str = "success"):
        self.actions.append((url, title, color))

    def __str__(self):
        return self.title

    def get_context(self):
        return {
            "title": self.title,
            "description": self.description,
            "actions": [
                {"url": action[0], "title": action[1], "color": action[2]}
                for action in self.actions
            ],
        }
