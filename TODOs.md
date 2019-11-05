# What do you want?

## Trainer
* `Trainer` handles a given period (epoch / certain iterations) of training + ecaluation
-[x] Merge Distributed Trainer into Trainer (distributed agnostic)
    * remove runner class
    
-[x] DDP has `_module`
-[ ] Iteration-based + epoch-based training / evaluation
-[ ] Easier training resume
    * move resume to `Task`

## Task
* `Task` handles a task from its beginning to the end
-[x] create Task class
-[ ] easier save/resume

````python
class VisionTask(Task):
    def data_loaders(self, key):...
````

# CLI
-[ ] Hydra and CLI tool

# Utils
-[ ] More containers

# If possible
-[ ] Easier reporters, callbacks