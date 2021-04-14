import collections
import datetime
import logging
import os
import time
import torch
import torch.utils.tensorboard
import numpy as np
from utils.metric_logger import MetricLogger


def do_train(model,train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs, checkpointer, arguments):
    start_training_time = time.time()
    end = time.time()
    logger = logging.getLogger("U.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()
    
    train_loss, valid_loss = [], []

    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join("outputs", 'tf_logs'))
    
    start_epoch = arguments["epoch"]
    
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
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 
                meters.update(total_loss=loss)
                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time)

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            
            # Adding early stopping (Gulle)
            if not phase=='train':
                if epoch_loss < lowest_val_loss:
                        lowest_val_loss = epoch_loss
                        es_stop_counter = 0

                if es_stop_counter >= 3:
                    print("Early stopping at epoch", epoch)
                    time_elapsed = time.time() - start
                    print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
                    return train_loss, valid_loss 

                es_stop_counter += 1
                
        """        
        if epoch % 2 == 0:
            eta_seconds = meters.time.global_avg * (epochs - epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            lr = optimizer.param_groups[0]['lr']
            to_log = [
                f"iter: {iteration:06d}",
                f"lr: {lr:.5f}",
                f'{meters}',
                f"eta: {eta_string}",
            ]
            if torch.cuda.is_available():
                mem = round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                to_log.append(f'mem: {mem}M')
            logger.info(meters.delimiter.join(to_log))
            global_step = iteration
            summary_writer.add_scalar(
                'losses/total_loss', loss, global_step=global_step)
            for loss_name, loss_item in loss_dict.items():
                summary_writer.add_scalar(
                    'losses/{}'.format(loss_name), loss_item,
                    global_step=global_step)
            summary_writer.add_scalar(
                'lr', optimizer.param_groups[0]['lr'],
                global_step=global_step)
        """
        if epoch % 2 == 0:
            checkpointer.save("model_{:06d}".format(epoch), **arguments)
        """
        if epoch % 3 == 0:
            eval_results = do_evaluation(cfg, model, iteration=iteration)
            for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                write_metric(
                    eval_result['metrics'], 'metrics/' + dataset,summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.
         """       
                
                

        time_elapsed = time.time() - start_training_time
        print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return train_loss, valid_loss    


