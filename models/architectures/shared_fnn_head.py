from models.architectures import baseline
from models import layers

class SharedFNNHead(baseline.BaseGPT):

    # only change the build transformer function?
    def build_transformer(self):
        transformers = super().build_transformer()
        # change the transformer to use the shared ffn head
        shared_ffn = transformers.h[0].mlp
        for block in transformers.h:
            block.mlp = shared_ffn
        return transformers