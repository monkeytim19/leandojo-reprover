from generator.model import RetrievalAugmentedGenerator
from retrieval.model import PremiseRetriever
from pytorch_lightning import Trainer
from huggingface_hub import snapshot_download
import torch

# def save_generator():
#     gen = RetrievalAugmentedGenerator(
#         model_name='kaiyuy/leandojo-lean4-tacgen-byt5-small',
#         lr=5e-4,
#         warmup_steps=2000,
#         num_beams=1,
#         eval_num_retrieved=100,
#         eval_num_workers=5,  # Lower this number if you don't have 80GB GPU memory.
#         eval_num_gpus=1,
#         eval_num_theorems=250,
#         max_inp_seq_len=2300,
#         max_oup_seq_len=512,
#         )
    
#     trainer = Trainer()
#     trainer.strategy.connect(gen)
#     trainer.save_checkpoint('my_files/generator.ckpt')

# def save_retriever():
#     ret = PremiseRetriever(
#         model_name="kaiyuy/leandojo-lean4-retriever-byt5-small",
#         lr=1e-4,
#         warmup_steps=2000,
#         max_seq_len=1024,
#         num_retrieved=100,
#     )

#     trainer = Trainer()
#     trainer.strategy.connect(ret)
#     trainer.save_checkpoint('my_files/retriever.ckpt')


# def save_reprover():
#     reprover = RetrievalAugmentedGenerator(
#         model_name='kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small',
#         lr=5e-4,
#         warmup_steps=2000,
#         num_beams=1,
#         eval_num_retrieved=100,
#         eval_num_workers=5,  # Lower this number if you don't have 80GB GPU memory.
#         eval_num_gpus=1,
#         eval_num_theorems=250,
#         max_inp_seq_len=2300,
#         max_oup_seq_len=512,
#         )
    
#     trainer = Trainer()
#     trainer.strategy.connect(reprover)
#     trainer.save_checkpoint('my_files/reprover.ckpt')

def save_generator_checkpoint(data_split):
    gen = RetrievalAugmentedGenerator(
        model_name='kaiyuy/leandojo-lean4-tacgen-byt5-small',
        lr=5e-4,
        warmup_steps=2000,
        num_beams=1,
        eval_num_retrieved=100,
        eval_num_workers=2,  # lowered from 5 to reduce GPU RAM load
        eval_num_gpus=1,
        eval_num_theorems=250,
        max_inp_seq_len=2300,
        max_oup_seq_len=512,
        )
    gen.load_state_dict(torch.load(f'my_files/downloaded_checkpoints/generator_{data_split}.ckpt/pytorch_model.bin'))
    gen.eval()
    trainer = Trainer()
    trainer.strategy.connect(gen)
    trainer.save_checkpoint(f'my_files/saved_checkpoints/generator_{data_split}.ckpt')

def save_retriever_checkpoint(data_split):
    ret = PremiseRetriever(
        model_name="kaiyuy/leandojo-lean4-retriever-byt5-small",
        lr=1e-4,
        warmup_steps=2000,
        max_seq_len=1024,
        num_retrieved=100,
    )
    ret.load_state_dict(torch.load(f'my_files/downloaded_checkpoints/retriever_{data_split}.ckpt/pytorch_model.bin'))
    ret.eval()
    trainer = Trainer()
    trainer.strategy.connect(ret)
    trainer.save_checkpoint(f'my_files/saved_checkpoints/retriever_{data_split}.ckpt')


def save_reprover_checkpoint(data_split):
    reprover = RetrievalAugmentedGenerator(
        model_name='kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small',
        lr=5e-4,
        warmup_steps=2000,
        num_beams=1,
        eval_num_retrieved=100,
        eval_num_workers=2,  # lowered from 5 to reduce GPU RAM load
        eval_num_gpus=1,
        eval_num_theorems=250,
        max_inp_seq_len=2300,
        max_oup_seq_len=512,
        ret_ckpt_path=f'my_files/saved_checkpoints/retriever_{data_split}.ckpt',
        )
    
    reprover.load_state_dict(torch.load(f'my_files/downloaded_checkpoints/reprover_{data_split}.ckpt/pytorch_model.bin'))
    reprover.eval()
    trainer = Trainer()
    trainer.strategy.connect(reprover)
    trainer.save_checkpoint(f'my_files/saved_checkpoints/reprover_{data_split}.ckpt')


def save_checkpoint():
    snapshot_download(repo_id="kaiyuy/leandojo-pl-ckpts", local_dir="my_files/downloaded_checkpoints")



if __name__ == "__main__":
    save_generator_checkpoint('novel_premises')
    save_retriever_checkpoint('novel_premises')
    save_reprover_checkpoint('novel_premises')
    pass
