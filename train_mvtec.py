from header import *
from datasets import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)# 可能用于分布式训练时的进程排名
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)             # 指定日志保存的路径
    # model configurations    模型配置
    parser.add_argument('--imagebind_ckpt_path', type=str)  # 存储ImageBind检查点的路径
    parser.add_argument('--vicuna_ckpt_path', type=str)     # 存储Vicuna检查点的路径
    parser.add_argument('--delta_ckpt_path', type=str)      # 在第一阶段训练的Delta参数的路径
    parser.add_argument('--max_tgt_len', type=int)          # 指定最大序列长度
    parser.add_argument('--stage', type=int)                # 指示训练阶段
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_root_path', type=str)

    return parser.parse_args()


# 初始化分布式训练环境
def initialize_distributed(args):
    # 获取环境变量MASTER_ADDR的值作为主节点的IP地址 如果环境变量不存在，则默认使用localhost
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    # 获取主节点的端口号，默认为'6000'
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    # 分布式训练参与节点的总数 未设置则默认为1
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    # 确定当前进程在分布式训练中的本地排名
    # 通过获取环境变量RANK的值，并将其模上当前机器上CUDA设备的数量来实现，确保local_rank在有效范围内
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    # 通过取args['local_rank']模上机器的CUDA设备总数来计算当前进程应使用的CUDA设备ID
    device = args['local_rank'] % torch.cuda.device_count()
    # 设置当前进程应使用的CUDA设备 始终为0
    torch.cuda.set_device(device)
    # 初始化分布式环境 nccl表示使用NVIDIA NCCL库作为后端进行分布式通信，这是进行分布式GPU训练时的常见选择
    # NVIDIA Collective Communications Library
    deepspeed.init_distributed(dist_backend='nccl')

# 设置随机数生成器的种子，以确保实验的可重复性
def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)    # 设置PyTorch在CPU模式下随机数生成器的种子
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)        # 设置PyTorch在CUDA模式下（针对单个GPU）随机数生成器的种子
        torch.cuda.manual_seed_all(seed)    # 设置PyTorch在CUDA模式下（针对所有GPU）随机数生成器的种子

# 配置和初始化深度学习训练环境
def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'

    # init中根据提供的参数加载一个配置文件
    config = load_config(args)
    # sh里args + base + openllama_peft(除去train)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):

    config_env(args)
    # 根据模型类型和训练阶段构建DeepSpeed配置文件的路径
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    # 使用DeepSpeed提供的HfDeepSpeedConfig类来加载DeepSpeed配置
    # 配置控制了DeepSpeed的许多方面，如模型并行度、优化器设置等
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])
    # 如果提供了日志路径，则使用logging.basicConfig设置日志记录
    # 包括日志的格式、级别、文件名（包含时间戳以避免覆盖）和模式

    #  指定了日志记录的格式，包括时间戳（%(asctime)s）、触发日志记录的源文件路径（%(pathname)s）、
    #  行号（%(lineno)d）、日志级别（%(levelname)s）和日志消息（%(message)s）
    # 设置了日志记录器捕捉的最低日志等级。在这里，它被设置为DEBUG，意味着所有等级为DEBUG及以上的日志都会被记录
    # DEBUG是最低的日志等级，因此这个设置会记录所有等级的日志。
    if args['log_path']:
        logging.basicConfig(
            # 2024-04-07 20:01:46,043 - /root/miniconda3/envs/yolov8/lib/python3.8/site-packages/PIL/PngImagePlugin.py
            # [line:190] - DEBUG: STREAM b'IHDR' 16 13
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            # train_Sun Apr  7 20_01_46 2024.log
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    # 根据传入的参数加载相应的数据集，并返回数据、迭代器和采样器
    train_data, train_iter, sampler = load_mvtec_dataset(args)
    train_data_sft, train_iter_sft, sampler = load_sft_dataset(args)

    # 根据epochs数、数据集长度、世界大小（分布式训练中的总GPU数）和DeepSpeed配置
    # （如每GPU的微批次大小和训练批次大小）计算训练所需的长度和总步数
    # 50*len(train_data)//1//1    3629
    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    # 2*50*len(train_data)//16    22681
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps

    # 根据提供的参数加载模型
    agent = load_model(args)
    # 在分布式训练中，这一步确保所有进程在继续执行之前都已到达这个点，以同步开始
    torch.distributed.barrier()

    # begin to train
    # 进度条的最大值被设置为2 * length
    pbar = tqdm(total=2 * length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        for batch, batch_sft in zip(train_iter, train_iter_sft):
            iter_every_epoch += 1
            # 调用agent.train_model方法进行训练。这个方法接受批次数据、当前步骤数current_step和进度条pbar作为参数
            # 每次训练后，对应的批次数据被删除（del batch和del batch_sft），以节省内存
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            del batch
            
            agent.train_model(
                batch_sft, 
                current_step=current_step, 
                pbar=pbar
            )
            del batch_sft
            current_step += 1
            # torch.cuda.empty_cache()
            # if iter_every_epoch % 1000 == 0:
            #     agent.save_model(args['save_path'], 0)


        # save at the end of the training
        # 每个epoch结束后，使用torch.distributed.barrier()同步不同进程（如果使用分布式训练的话），然后保存模型
        torch.distributed.barrier()
        agent.save_model(args['save_path'], 0)

if __name__ == "__main__":
    args = parser_args()
    # argparse.Namespace对象（即 parse_args() 返回的对象），vars() 函数会将这个命名空间对象转换成一个字典
    args = vars(args)
    args['layers'] = [7, 15, 23, 31]
    main(**args)
