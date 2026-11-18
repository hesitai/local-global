import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='global batch size across all GPUs')
        parser.add_argument(
            '--max_iters', type=int, default=0,
            help='total training batches; overrides nepoch when positive (0: epoch mode)')
        parser.add_argument(
            '--warmup_iters', type=int, default=2_820,
            help='linear warmup batches with --max_iters and --warmup (0: no warmup)')
        parser.add_argument('--nepoch', type=int, default=1500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument(
            '--train_drop_last', action='store_true', default=False,
            help='drop the final incomplete global training batch each epoch')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='SIDD')
        parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer_B/models/model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument(
            '--lr_range_test', action='store_true', default=False,
            help='run a short exponential learning-rate range test and exit')
        parser.add_argument(
            '--lr_range_min', type=float, default=1e-5,
            help='minimum LR used by --lr_range_test')
        parser.add_argument(
            '--lr_range_max', type=float, default=1e-3,
            help='maximum LR used by --lr_range_test')
        parser.add_argument(
            '--lr_range_iters', type=int, default=2000,
            help='maximum optimizer iterations in --lr_range_test')
        parser.add_argument(
            '--lr_range_smooth_beta', type=float, default=0.98,
            help='EMA beta used to smooth the LR-range training loss')
        parser.add_argument(
            '--lr_range_diverge_factor', type=float, default=4.0,
            help='stop when smoothed loss exceeds this multiple of its best value')
        parser.add_argument(
            '--lr_range_burnin', type=int, default=100,
            help='minimum iterations before divergence-based early stopping')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='comma-separated CUDA device IDs')
        parser.add_argument('--arch', type=str, default ='Uformer_B',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
        parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')
        parser.add_argument('--use_local_branch', action='store_true', default=False,
                            help='enable Omni DWConv-PWConv-PWConv-DWConv branch for attention K/V')
        parser.add_argument('--double_process', action='store_true', default=False,
                            help='run a second attention+FFN process only in the last block of each encoder/decoder stage')
        parser.add_argument('--local_expansion', type=int, default=2,
                            help='channel reduction ratio inside the Omni local branch')
        parser.add_argument('--use_channel_aware', action='store_true', default=False,
                            help='gate attention outputs with DiNAT-IR CAM')
        parser.add_argument('--use_ddem', action='store_true', default=False,
                            help='enable MODEM-style degradation priors for Q/K guidance')
        parser.add_argument('--ddem_stage', type=int, choices=[1, 2], default=2,
                            help='1: teacher uses blur+GT; 2: student uses blur only')
        parser.add_argument(
            '--ddem_scratch_student', action='store_true', default=False,
            help=(
                'train a stage-2 Blur-only DDEM from random initialization '
                'without loading/creating a teacher or computing KD loss'))
        parser.add_argument('--ddem_dim', type=int, default=32,
                            help='shared Z0/Z1 prior dimension before each stage-specific adapter')
        parser.add_argument('--use_kernel_response', action='store_true', default=False)
        parser.add_argument('--kernel_response_groups', type=int, choices=[1, 4], default=1,
                            help='1: previous response model; 4: grouped expert gates')
        parser.add_argument(
            '--kernel_response_variant', choices=['serial', 'joint_residual'],
            default='serial',
            help=('response controller architecture; serial preserves existing checkpoints; '
                  'the online training launcher selects joint_residual by default'))
        parser.add_argument('--response_gate_log_interval', type=int, default=0,
                            help='log first training batch gate statistics every N epochs; 0 disables')
        parser.add_argument('--response_loss_weight', type=float, default=0.01)
        parser.add_argument('--response_loss_mode', choices=['complex', 'magnitude'], default='complex')
        parser.add_argument('--ddem_blocks', type=int, default=2,
                            help='number of DDEM feature blocks')
        parser.add_argument('--ddem_backbone', type=str,
                            choices=['local_global', 'transformer', 'conv'],
                            default='local_global',
                            help='DDEM extractor: proposed local-global base blocks, standard LeWin, or legacy CNN')
        parser.add_argument('--ddem_heads', type=int, default=1,
                            help='attention heads inside Transformer DDEM; 1 matches Uformer dim=32 stage')
        parser.add_argument('--ddem_win_size', type=int, default=8,
                            help='window size inside Transformer DDEM')
        parser.add_argument('--ddem_stage1_weights', type=str, default='',
                            help='stage-1 checkpoint used to initialize/freeze the stage-2 teacher')
        parser.add_argument('--ddem_kd_weight', type=float, default=1.0,
                            help='stage-2 prior feature distillation loss weight')
        parser.add_argument('--ddem_temperature', type=float, default=0.15,
                            help='temperature for MODEM-style KL distillation')
        parser.add_argument('--kd_grad_stat_interval', type=int, default=110,
                            help='report DDEM restoration/KD gradient statistics every N training iterations; <=0 disables')
        parser.add_argument('--grad_clip', type=float, default=1.0,
                            help='global gradient-norm limit for DDEM training; <=0 disables clipping')
        parser.add_argument('--init_weights', type=str, default='',
                            help='model-only initialization checkpoint; unlike --resume, optimizer/epoch are not loaded')
        parser.add_argument('--adapter_warmup_epochs', type=int, default=0,
                            help='freeze restoration backbone for this many initial epochs and train only DDEM/adapters')
        parser.add_argument('--backbone_lr_scale', type=float, default=0.1,
                            help='restoration-backbone LR divided by DDEM/adapter LR during DDEM training')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
        parser.add_argument(
            '--val_epoch_interval', type=int, default=1,
            help='run validation once every N epochs; values below 1 are treated as 1')
        parser.add_argument('--disable_validation', action='store_true', default=False,
                            help='skip initial and periodic validation during training')
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--ema', action='store_true', help='EMA from epoch 4, decay 0.999')
        parser.add_argument('--train_dir', type=str, default ='./datasets/SIDD/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='./datasets/SIDD/val',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        # Online single-image 2D blur synthesis. A separate CPU process pool
        # periodically generates aligned fixed-size Blur/GT candidate patches;
        # each epoch samples one cached candidate per GT.
        student_root = os.path.dirname(os.path.abspath(__file__))
        joint2d_root = os.path.dirname(student_root)
        default_motion_project = os.path.join(
            joint2d_root, 'motion-flow-syn-python-flat')
        default_uav_dataset = os.path.join(
            student_root, 'dataset/UAV_single_image_pose_split')
        parser.add_argument(
            '--online_2d', action='store_true', default=False,
            help='train from periodically refreshed single-image blur pools')
        parser.add_argument(
            '--online_2d_project', type=str, default=default_motion_project,
            help='motion-flow-syn-python-flat project root')
        parser.add_argument(
            '--online_2d_python', type=str, default='',
            help='generator Python; empty uses <online_2d_project>/.venv/bin/python')
        parser.add_argument(
            '--online_2d_gt_root', type=str,
            default=os.path.join(default_uav_dataset, 'train/sharp'),
            help='directory containing immutable full-resolution sharp GT images')
        parser.add_argument(
            '--online_2d_cache_root', type=str,
            default=os.path.join(student_root, 'online_2d_cache'),
            help='refresh-block Blur/GT candidate pool cache')
        parser.add_argument('--online_2d_workers', type=int, default=16)
        parser.add_argument(
            '--online_2d_backend', choices=['cuda', 'cpu'], default='cpu',
            help='CUDA uses batched PyTorch grid_sample; CPU uses OpenCV')
        parser.add_argument(
            '--online_2d_gpus', type=str, default='',
            help='physical GPU IDs for generation; empty reuses --gpu')
        parser.add_argument(
            '--online_2d_cuda_subframe_batch', type=int, default=32,
            help='number of exposure subframes warped together per CUDA batch')
        parser.add_argument('--online_2d_seed', type=int, default=1234)
        parser.add_argument('--online_2d_cases', type=str, default='all')
        parser.add_argument('--online_2d_strength_min', type=float, default=1.0)
        parser.add_argument('--online_2d_strength_max', type=float, default=1.80)
        parser.add_argument(
            '--online_2d_variants_per_gt', type=int, default=12,
            help='blur candidates generated per GT in each reusable cache pool')
        parser.add_argument(
            '--online_2d_pool_refresh_epochs', type=int, default=50,
            help='number of training epochs that reuse one generated blur pool')
        parser.add_argument(
            '--online_2d_dataset_scope', choices=['quarter', 'full', 'fifth'],
            default='quarter',
            help=(
                'experiment dataset label: quarter uses sorted indices 0,4,8,...; '
                'fifth identifies the fixed 1/5 subset; full uses all GT. When '
                '--online_2d_preselected_gt_root is set, the given GT directory '
                'is always consumed in full.'
            ))
        parser.add_argument(
            '--online_2d_preselected_gt_root', action='store_true', default=False,
            help='the GT directory already contains the exact fixed source set')
        parser.add_argument('--online_2d_fixed_subframes', type=int, default=31)
        parser.add_argument(
            '--online_2d_adaptive_subframes', action='store_true', default=True,
            help='adapt temporal samples to motion distance/frequency (default)')
        parser.add_argument(
            '--online_2d_fixed_sampling', action='store_false',
            dest='online_2d_adaptive_subframes',
            help='use the fixed odd --online_2d_fixed_subframes count')
        parser.add_argument('--online_2d_interpolation_margin', type=int, default=2)
        parser.add_argument('--online_2d_crop_trials', type=int, default=256)
        parser.add_argument(
            '--online_2d_max_images', type=int, default=0,
            help='debug limit; 0 uses every sharp training GT')
        parser.add_argument(
            '--online_2d_keep_cache', action='store_true', default=False,
            help='keep consumed epoch caches for debugging instead of deleting them')

        # ddp
        parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')#不需要赋值，启动命令 torch.distributed.launch会自动赋值
        parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
        parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
        return parser
