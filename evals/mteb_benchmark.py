"""Run the mteb benchmark to evaluate the model embeddings.
https://arxiv.org/abs/2210.07316"""

import mteb

import benchmark


class MTEBBenchmark(benchmark.Benchmark):
    def __init__(self, name, model, description=""):
        self.name = name
        self.description = description
        self.model = model
        self.metrics = {}

    def execute(self):
        mteb.MTEB(
            tasks=["Banking77Classification", "RedditClustering", "SummEval"]
        ).run(model=model)
        return {}

if __name__ == "__main__":
    mteb_model = mteb.MTEB(
        tasks=["Banking77Classification", "RedditClustering", "SummEval"]
    )
    mteb_model.run(model=benchmark.FauxModel())
