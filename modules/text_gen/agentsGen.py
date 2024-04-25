# IMPORTS
# ----------------------------------


# CLASS DEVELOPMENT
# ----------------------------------
class AgentGen:
    def __init__(self, model, etc):
        self.model = model
        self.etc = etc
        self.knowledge = Knowledge()