import matplotlib.pyplot as plt
import matplotlib.animation as animation

from json import load

def data_generator():
    data = load(open('output/small_kw.json'))

    n = len(data['fcfs']['observation'])

    POSITIONS = len(data['fcfs']['heights'][0])
    FPS = 60
    acc_reward_fcfs = 0
    acc_reward_ddpg = 0

    previous_heights_fcfs = data['fcfs']['heights'][0]
    previous_heights_ddpg = data['ddpg']['heights'][0]

    all_acc_rewards_fcfs = []
    all_acc_rewards_ddpg = []

    steps = []

    for i in range(n):
        heights_fcfs = data['fcfs']['heights'][i]
        heights_ddpg = data['ddpg']['heights'][i]

        reward_fcfs = data['fcfs']['reward'][i]
        reward_ddpg = data['ddpg']['reward'][i]

        for t in range(1, FPS+1):
            steps.append(i+(t/FPS))

            # HEIGHTS
            tmp_heights_fcfs = [0] * POSITIONS
            tmp_heights_ddpg = [0] * POSITIONS

            for pos in range(POSITIONS):
                tmp_heights_fcfs[pos] = previous_heights_fcfs[pos] + ((heights_fcfs[pos] - previous_heights_fcfs[pos])/FPS) * t
                tmp_heights_ddpg[pos] = previous_heights_ddpg[pos] + ((heights_ddpg[pos] - previous_heights_ddpg[pos])/FPS) * t

            # REWARDS
            tmp_acc_reward_fcfs = acc_reward_fcfs + (reward_fcfs/FPS)*t
            tmp_acc_reward_ddpg = acc_reward_ddpg + (reward_ddpg/FPS)*t

            all_acc_rewards_fcfs.append(tmp_acc_reward_fcfs)
            all_acc_rewards_ddpg.append(tmp_acc_reward_ddpg)

            yield tmp_heights_fcfs, tmp_heights_ddpg, all_acc_rewards_fcfs, all_acc_rewards_ddpg, steps

        previous_heights_fcfs = heights_fcfs
        previous_heights_ddpg = heights_ddpg

        acc_reward_fcfs += reward_fcfs
        acc_reward_ddpg += reward_ddpg

if __name__ == '__main__':
    data = load(open('output/small_kw.json'))
    n = len(data['fcfs']['observation'])
    POSITIONS = len(data['fcfs']['heights'][0])

    fig, axes = plt.subplots(1, 2, num=1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    plt.subplots_adjust(top = 0.9, bottom = 0.1, wspace = 0.25)
    ax1, ax2 = axes

    bars = []

    ids = list(range(20))

    bar_width = 0.4
    n_bars = POSITIONS

    offset = bar_width / 2
    bar_obj = ax1.bar([i - offset for i in ids],
                      [0]*n_bars,
                      width=bar_width,
                      label='fcfs')
    bars.append(bar_obj)
    
    bar_obj = ax1.bar([i + offset for i in ids],
                      [0]*n_bars,
                      width=bar_width,
                      label='ddpg')
    bars.append(bar_obj)


    lines = []
    for name in ['fcfs', 'ddpg']:
        line_obj, = ax2.plot([], [], lw=2, label=name)
        lines.append(line_obj)
    
        
    ax1.set_ylim(0, 109)
    ax1.set_xlim(-0.5, 20.5)

    ax2.set_ylim(0, 1e4)
    ax2.set_xlim(0, 109)

    ax1.set_xlabel('Position')
    ax1.set_ylabel('Current Capacity in %')
    ax1.set_title('Comparison current state')
    ax1.legend(loc='upper right')

    ax2.set_ylabel('Reward')
    ax2.set_xlabel('Step')
    ax2.set_title('Comparison reward')
    ax2.legend(loc='upper right')

    total_width = 0.8
    single_width = 1
    n_bars = 20
    bar_width = total_width / n_bars

    def visualize(data):
        heights_fcfs, heights_ddpg, acc_reward_history_fcfs, acc_reward_history_ddpg, steps = data

        ax2.set_ylim(0, max(0.01, max(max(acc_reward_history_fcfs), max(acc_reward_history_ddpg)) * 1.2))
        ax2.set_xlim(0, max(steps) + 10)
        ax2.figure.canvas.draw()

        lines[0].set_data(steps, acc_reward_history_fcfs)
        lines[1].set_data(steps, acc_reward_history_ddpg)


        for i, bar in enumerate(bars[0]):
            bar.set_height(heights_fcfs[i])
            
        for i, bar in enumerate(bars[1]):
            bar.set_height(heights_ddpg[i])

        return lines[0], lines[1]
    
    plt.rcParams['animation.ffmpeg_path'] ='C:\\Users\\Qunor\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'
    ani = animation.FuncAnimation(fig,
                                  visualize,
                                  data_generator,
                                  blit=True,
                                  interval=1000,
                                  repeat=False,
                                  save_count=5000)
    ani.save('output/comparison.mp4', writer=animation.FFMpegWriter(fps=60))