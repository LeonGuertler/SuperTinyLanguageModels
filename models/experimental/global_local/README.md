## TODO 
- dual dataloader for delimitations [focus on loading the same number of chunks, rather than same number of bytes]

- Core model
    - token projection from global to local with weight tying 
    - alternate between byte-level sliding window attention and global attention blocks



Thoughts:
- probably makes sense the wrap the down projected global token into two special tokens (or two learned embeddings anyway)