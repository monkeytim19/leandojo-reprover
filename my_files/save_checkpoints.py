from generator.model import RetrievalAugmentedGenerator
from retrieval.model import PremiseRetriever
from pytorch_lightning import Trainer

def save_generator():
    gen = RetrievalAugmentedGenerator(
        model_name='kaiyuy/leandojo-lean4-tacgen-byt5-small',
        lr=5e-4,
        warmup_steps=2000,
        num_beams=1,
        eval_num_retrieved=100,
        eval_num_workers=5,  # Lower this number if you don't have 80GB GPU memory.
        eval_num_gpus=1,
        eval_num_theorems=250,
        max_inp_seq_len=2300,
        max_oup_seq_len=512,
        )
    
    trainer = Trainer()
    trainer.strategy.connect(gen)
    trainer.save_checkpoint('my_files/generator.ckpt')


def save_retriever():
    ret = PremiseRetriever(
        model_name="kaiyuy/leandojo-lean4-retriever-byt5-small",
        lr=1e-4,
        warmup_steps=2000,
        max_seq_len=1024,
        num_retrieved=100,
    )

    trainer = Trainer()
    trainer.strategy.connect(ret)
    trainer.save_checkpoint('my_files/retriever.ckpt')


def save_reprover():
    reprover = RetrievalAugmentedGenerator(
        model_name='kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small',
        lr=5e-4,
        warmup_steps=2000,
        num_beams=1,
        eval_num_retrieved=100,
        eval_num_workers=5,  # Lower this number if you don't have 80GB GPU memory.
        eval_num_gpus=1,
        eval_num_theorems=250,
        max_inp_seq_len=2300,
        max_oup_seq_len=512,
        )
    
    trainer = Trainer()
    trainer.strategy.connect(reprover)
    trainer.save_checkpoint('my_files/reprover.ckpt')


if __name__ == "__main__":
    save_generator()
    pass
