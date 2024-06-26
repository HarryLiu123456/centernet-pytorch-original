'''
这段代码是一个训练和验证过程的函数，主要用于训练和验证基于CenterNet的目标检测模型。
它包含了对模型进行训练和验证的主要步骤，包括前向传播、计算损失、反向传播、优化器更新、
以及在验证集上的验证。此外，它还包含了一些用于监控训练过程的工具，如进度条（tqdm）和学习率监控。
'''
import os

import torch
from net.centernet_training import focal_loss, reg_l1_loss
from tqdm import tqdm

from util.utils import get_lr


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, scaler, backbone, save_period, save_dir, local_rank=0):
    total_r_loss    = 0
    total_c_loss    = 0
    total_loss      = 0
    val_loss        = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()

        if backbone=="resnet50":
            hm, wh, offset  = model_train(batch_images)
            c_loss          = focal_loss(hm, batch_hms)
            wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
            
            loss            = c_loss + wh_loss + off_loss

            total_loss      += loss.item()
            total_c_loss    += c_loss.item()
            total_r_loss    += wh_loss.item() + off_loss.item()
        else:
            outputs         = model_train(batch_images)
            loss            = 0
            c_loss_all      = 0
            r_loss_all      = 0
            index           = 0
            for output in outputs:
                hm, wh, offset = output["hm"].sigmoid(), output["wh"], output["reg"]
                c_loss      = focal_loss(hm, batch_hms)
                wh_loss     = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss    = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss        += c_loss + wh_loss + off_loss
                
                c_loss_all  += c_loss
                r_loss_all  += wh_loss + off_loss
                index       += 1
            total_loss      += loss.item() / index
            total_c_loss    += c_loss_all.item() / index
            total_r_loss    += r_loss_all.item() / index
        loss.backward()
        optimizer.step()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_r_loss'  : total_r_loss / (iteration + 1), 
                                'total_c_loss'  : total_c_loss / (iteration + 1),
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
            
        with torch.no_grad():
            if cuda:
                batch = [ann.cuda(local_rank) for ann in batch]
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

            if backbone=="resnet50":
                hm, wh, offset  = model_train(batch_images)
                c_loss          = focal_loss(hm, batch_hms)
                wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                loss            = c_loss + wh_loss + off_loss

                val_loss        += loss.item()
            else:
                outputs = model_train(batch_images)
                index   = 0
                loss    = 0
                for output in outputs:
                    hm, wh, offset  = output["hm"].sigmoid(), output["wh"], output["reg"]
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            += c_loss + wh_loss + off_loss
                    index           += 1
                val_loss            += loss.item() / index

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))