import collections
import datetime
import logging
import os
import time
import torch
import torch.utils.tensorboard
import numpy as np
from configs import cfg


def do_train(model,train_dl, valid_dl, loss_fn, optimizer, dice_fn, acc_fn, epochs, checkpointer, arguments):
    start_training_time = time.time()
    end = time.time()
    logger = logging.getLogger("U.trainer")
    logger.info("Start training ...")
    
    
    
    #load in data
    start_epoch = arguments["epoch"] 
    train_loss = arguments["train_loss"]
    valid_loss = arguments["valid_loss"]
    valid_dice = arguments["valid_dice"]
    pixel_acc = arguments["pixel_acc"]
    
    best_acc = 0.0
    lowest_val_loss = np.inf
    es_stop_counter = 0
    for epoch in range(start_epoch,epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_dice = 0.0
            running_acc = 0.0
            
            step = 0
            
            ####
            arguments["epoch"] = epoch
              ####  
            
            
            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1
                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                dice = dice_fn(outputs, y)
                acc = acc_fn(outputs, y)
                
                running_dice  += dice*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                running_acc  += acc*dataloader.batch_size
                
                batch_time = time.time() - end
                end = time.time()
                    
                

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Dice: {}, Acc: {}'.format(phase, epoch_loss, epoch_dice, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            if phase=='valid':
                valid_dice.append(epoch_dice)
                pixel_acc.append(epoch_acc)
                arguments["train_loss"] = train_loss
                arguments["valid_loss"] = valid_loss
                arguments["valid_dice"] = valid_dice
                arguments["pixel_acc"] = pixel_acc

            
            # Added early stopping (Gulle)
            if not phase=='train':
                if epoch_loss < lowest_val_loss:
                        lowest_val_loss = epoch_loss
                        es_stop_counter = 0

                if es_stop_counter >= cfg.TRAINING.EARLY_STOP_COUNT: 
                    print("Early stopping at epoch", epoch)
                    time_elapsed = time.time() - start_training_time
                    print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
                    return train_loss, valid_loss, valid_dice, pixel_acc  

                es_stop_counter += 1

        
        ### save model every second epoch
        if epoch % 2 == 0:
            checkpointer.save("model_{:06d}".format(epoch), **arguments)  
        

        time_elapsed = time.time() - start_training_time
        print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / epochs))
    return train_loss, valid_loss, valid_dice, pixel_acc    
