from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def callback_warpper(log_name, args):
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename=f'best_model-{log_name}',
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor]
    return callbacks
