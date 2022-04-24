import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import datasets
from pytorch_ie.models.transformer_span_classification import TransformerSpanClassificationModel
from pytorch_ie.taskmodules.transformer_span_classification import (
    TransformerSpanClassificationTaskModule,
)


def main():
    pl.seed_everything(42)

    model_output_path = "./model_output/"
    model_name = "bert-base-cased"
    num_epochs = 10
    batch_size = 32

    dataset = datasets.load_dataset(
        path="pie/conll2003",
    )
    train_docs, val_docs = list(dataset["train"]), list(dataset["validation"])

    print("train docs: ", len(train_docs))
    print("val docs: ", len(val_docs))

    task_module = TransformerSpanClassificationTaskModule(
        tokenizer_name_or_path=model_name,
        max_length=128,
    )

    task_module.prepare(train_docs)

    train_dataset = task_module.encode(train_docs, encode_target=True)
    val_dataset = task_module.encode(val_docs, encode_target=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_module.collate,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_module.collate,
    )

    model = TransformerSpanClassificationModel(
        model_name_or_path=model_name,
        num_classes=len(task_module.label_to_id),
        t_total=len(train_dataloader) * num_epochs,
        learning_rate=1e-4,
    )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val/f1",
    #     dirpath=model_output_path,
    #     filename="zs-ner-{epoch:02d}-val_f1-{val/f1:.2f}",
    #     save_top_k=1,
    #     mode="max",
    #     auto_insert_metric_name=False,
    #     save_weights_only=True,
    # )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=num_epochs,
        gpus=0,
        checkpoint_callback=False,
        # callbacks=[checkpoint_callback],
        precision=32,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # task_module.save_pretrained(model_output_path)

    # trainer.save_checkpoint(model_output_path + "model.ckpt")
    # or
    # model.save_pretrained(model_output_path)


if __name__ == "__main__":
    main()
