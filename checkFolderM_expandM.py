import argparse
from pylab import *
import os
import audio_utilities
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt

def getM(txtdir):
    '''
    to get M feature map
    :param txtdir: str, lastm.txt, in K*T shape
    :return: dimMeans(K-dim),frameMeans(T-dim)
    '''

    norm = np.loadtxt(txtdir, comments=['m', '[', '#', 'l'])
    norm = norm.T #T*K

    # print(np.mean(norm[:int(norm.shape[0] / 2), :]))这是为了看前后两部分音乐m的值变化
    # print(np.mean(norm[int(norm.shape[0] / 2):, :]))

    plt.figure(figsize=(8, 6))
    plt.imshow(norm, cmap='hot', aspect='auto', vmax=3 * np.mean(norm))
    plt.colorbar()
    plt.title('m-norm heatmap')
    plt.ylabel('%d frames' % norm.shape[0])
    plt.xlabel('%d dim of a vector' % norm.shape[1])
    # plt.text(200, -10, 'H=5.6666, D=0.555' )
    plt.text(-30,-10, 'μ=%.3e, δ=%.3e' % (np.mean(norm), np.var(norm)))
    plt.savefig(os.path.dirname(txtdir) + '/m.png')
    plt.clf()

    dimMeans = np.mean(norm, axis=0)
    frameMeans = np.mean(norm, axis=1)

    plt.figure()
    plt.hist(dimMeans, bins=100, normed=False)
    plt.title('hist for avg-m in each dim')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(dimMeans), np.var(dimMeans)))
    plt.savefig(os.path.dirname(txtdir) + '/avgm-ondim.png')
    plt.clf()

    plt.figure()
    plt.hist(norm.flatten(), bins=100, normed=False)
    plt.title('hist for avg-m in each point')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(norm), np.var(norm)))
    plt.savefig(os.path.dirname(txtdir) + '/avgm-onpoint.png')
    plt.clf()

    print('max of m is ',end='')
    print(norm.max())

    plt.figure()
    plt.hist(frameMeans, bins=100, normed=False)
    plt.title('hist for avg-m in each frame')
    plt.text(-30, -10, 'μ=%.3e, δ=%.3e' % (np.mean(frameMeans), np.var(frameMeans)))
    plt.savefig(os.path.dirname(txtdir) + '/avgm-onframe.png')
    plt.clf()

    return dimMeans, frameMeans, norm

def saveMusic(stft_modified_scaled,fftangle,outdir):
    # Author: Brian K. Vogel
    # brian.vogel@gmail.com
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default="000000000.wav",
                        help='Input WAV file')
    parser.add_argument('--sample_rate_hz', default=16000, type=int,
                        help='Sample rate in Hz')
    parser.add_argument('--fft_size', default=512, type=int,
                        help='FFT siz')
    parser.add_argument('--iterations', default=300, type=int,
                        help='Number of iterations to run')
    parser.add_argument('--enable_filter', action='store_true',
                        help='Apply a low-pass filter')
    parser.add_argument('--enable_mel_scale', action='store_true',
                        help='Convert to mel scale and back')
    parser.add_argument('--cutoff_freq', type=int, default=1000,
                        help='If filter is enable, the low-pass cutoff frequency in Hz')
    args = parser.parse_args()

    hopsamp = 160   # stride 16000 Hz x 10 ms
    proposal_spectrogram = stft_modified_scaled * np.exp(1.0j * fftangle)
    #ejw=cosw+jsinw
    x_reconstruct = audio_utilities.istft_for_reconstruction(proposal_spectrogram, args.fft_size, hopsamp)


    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:x_reconstruct = x_reconstruct / max_sample

    audio_utilities.save_audio_to_flac(x_reconstruct, args.sample_rate_hz,outfile=outdir)

def makeVoice(prefftdir, dimMeans, frameMeans, norm, dimThreshes=None, frameThreshes=None):
    '''

    :param prefftdir:str, preFft.txt
    :param dimMeans: got from getM()
    :param frameMeans: framehist got from getM()
    :param dimThreshes:list, default=[0.1,0.3,0.8]
    :param frameThreshes:list, default=[0.1,0.3,0.5]
    :return:
    '''

    dimhist = np.sort(dimMeans)
    framehist = np.sort(frameMeans)
    normhist=np.sort(norm.flatten())

    # normThreshes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    normThreshes = [0.2,0.5,0.8]

    fft = np.loadtxt(prefftdir, comments=['p', '[', '#', 'l'])
    fft = fft.T

    fftnorm = np.zeros((fft.shape[0], int(fft.shape[1] / 2)))
    fftangle = np.zeros((fft.shape[0], int(fft.shape[1] / 2)))

    for i in range(0, fft.shape[0]):
        frame = fft[i]
        real = frame[0::2]
        imag = frame[1::2]
        comp2 = np.square(real) + np.square(imag)
        fftnorm[i] = np.sqrt(comp2)
        fftangle[i] = np.angle(real+imag*1.0j)
    fftnorm /= 32768.0

    outdir = os.path.dirname(prefftdir) + '/original.flac'
    saveMusic(fftnorm, fftangle, outdir)

    plt.figure()
    plt.imshow(fftnorm ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('original fft')
    plt.ylabel('time index')
    plt.xlabel('frequency bin index')
    plt.savefig(outdir.replace('flac', 'png'), dpi=150)
    plt.clf()

    for threshPercent in normThreshes:
        thresh = normhist[-1 * int(len(normhist) * threshPercent)]
        outIndexs = [(i,j) for i in range(norm.shape[0]) for j in range (norm.shape[1]) if norm[i][j] > thresh]
        outIndex_rev = [(i,j) for i in range(norm.shape[0]) for j in range(norm.shape[1]) if norm[i][j] <= thresh]
        #here is in T*K format

        newnorm = np.copy(fftnorm)
        newnorm_rev = np.copy(fftnorm)
        zero_list = []
        zero_list_rev = []
        outdir = os.path.dirname(prefftdir) + '/remove_%d' % (100 * threshPercent) + 'high.flac'
        outdir_rev = os.path.dirname(prefftdir) + '/remove_%d' % (100 - 100 * threshPercent) + 'low.flac'
        # eg. 20high.wav表示20%的高点置零，20low同理

        for ind in outIndexs:
            newnorm[ind] = 0
            i,j=ind
            zero_list.append(2*j)
            zero_list.append(i)
            zero_list.append(2*j+1)
            zero_list.append(i)
        np.savetxt(outdir.replace('flac', 'txt'), zero_list, fmt="%d")

        for ind in outIndex_rev:
            newnorm_rev[ind] = 0
            i, j = ind
            zero_list_rev.append(2 * j)
            zero_list_rev.append(i)
            zero_list_rev.append(2 * j + 1)
            zero_list_rev.append(i)
        np.savetxt(outdir_rev.replace('flac', 'txt'), zero_list_rev, fmt="%d")

        # npoutdir=outdir.replace('flac','txt')
        # npoutdir_rev=outdir_rev.replace('flac','txt')
        # np.savetxt(npoutdir,newnorm)
        # np.savetxt(npoutdir_rev, newnorm_rev)

        saveMusic(newnorm,fftangle, outdir)
        saveMusic(newnorm_rev,fftangle,outdir_rev)

        stft_mag = newnorm.T
        plt.figure()
        plt.imshow(stft_mag.T ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title('%d%% on high' % (100 * threshPercent))
        plt.ylabel('time index')
        plt.xlabel('frequency bin index')
        plt.savefig(os.path.dirname(prefftdir) + '/remove_%d' % (100 * threshPercent) + 'high.png', dpi=150)
        plt.clf()

        stft_mag = newnorm_rev.T
        plt.figure()
        plt.imshow(stft_mag.T ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title('%d%% on low' % (100-100 * threshPercent))
        plt.ylabel('time index')
        plt.xlabel('frequency bin index')
        plt.savefig(os.path.dirname(prefftdir) + '/remove_%d' % (100-100 * threshPercent) + 'low.png', dpi=150)
        plt.clf()

    return fftangle

def makeVoicefor_mask(maskdirs, ismask_KT=False, fftangle=None):
    '''
    generate mask music
    :param maskdirs: list of string
    :param fftangle: np, [T K]
    :param ismask_KT: bool, is False, maskdir is 2K*T, and if not, we'll need fftangle
    :return:
    '''
    if ismask_KT:
        for maskdir in maskdirs:
            music_mask_norm = np.loadtxt(maskdir, comments=['m', '[', '#', 'l'])
            music_mask_norm = music_mask_norm.T
            music_mask_norm /= 32768.0

            plt.figure()
            plt.imshow(music_mask_norm ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.title(os.path.basename(maskdir))
            plt.ylabel('time index')
            plt.xlabel('frequency bin index')
            plt.savefig(maskdir.replace('txt', 'png'), dpi=150)
            plt.clf()

            outdir = maskdir.replace('txt','flac')
            saveMusic(music_mask_norm, fftangle, outdir)

    else:
        for maskdir in maskdirs:
            music_mask = np.loadtxt(maskdir, comments=['m', '[', '#', 'l'])
            music_mask = music_mask.T

            music_mask_norm = np.zeros((music_mask.shape[0], int(music_mask.shape[1] / 2)))
            music_mask_angle = np.zeros((music_mask.shape[0], int(music_mask.shape[1] / 2)))

            for i in range(0, music_mask.shape[0]):
                frame = music_mask[i]
                real = frame[0::2]
                imag = frame[1::2]
                comp2 = np.square(real) + np.square(imag)
                music_mask_norm[i] = np.sqrt(comp2)
                music_mask_angle[i] = np.angle(real + imag * 1.0j)
            music_mask_norm /= 32768.0

            plt.figure()
            plt.imshow(music_mask_norm ** 0.125, cmap='hot', aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.title(os.path.basename(maskdir))
            plt.ylabel('time index')
            plt.xlabel('frequency bin index')
            plt.savefig(maskdir.replace('txt', 'png'), dpi=150)
            plt.clf()

            outdir = maskdir.replace('txt', 'flac')
            saveMusic(music_mask_norm, music_mask_angle, outdir)

def getDft(src,outdir):
    fft=np.load(src)
    print(fft.shape)
    fftnorm = np.zeros((int(fft.shape[0]/2),fft.shape[1]))
    for i in range(0, fft.shape[1]):
        frame = fft[:,i]
        real = frame[0::2]
        imag = frame[1::2]
        comp2 = np.square(real) + np.square(imag)
        fftnorm[:,i] = np.sqrt(comp2)
    fftnorm=fftnorm.T
    mean=np.mean(fftnorm)
    stdvar=np.sqrt(np.var(fftnorm))
    for i in range(fftnorm.shape[0]):
        for j in range(fftnorm.shape[1]):
            fftnorm[i][j]=(fftnorm[i][j]-mean)/stdvar

    name=os.path.basename(src)
    rawname=os.path.splitext(name)[0]+'dft'
    np.save(outdir+rawname,fftnorm)


if __name__ == '__main__':
    wholeDir = '/Users/markdana/Desktop/data'
    for folder in os.listdir(wholeDir):
        if folder in ['.DS_Store','000000000.flac','000000000.tkn']:continue

        files = os.listdir(os.path.join(wholeDir, folder))
        lossdir = os.path.join(wholeDir, folder, 'loss.txt')
        ys=np.loadtxt(lossdir)

        plt.plot(np.arange(len(ys)), ys)
        plt.savefig(os.path.dirname(lossdir) + '/loss.png', dpi=150)
        plt.clf()

        txtdir = os.path.join(wholeDir, folder, 'lastm.txt')
        prefftdir = os.path.join(wholeDir, folder, 'preFft.txt')
        dimMeans, frameMeans, norm = getM(txtdir)

        T,K=norm.shape[0],norm.shape[1]

        fftangle = makeVoice(prefftdir, dimMeans, frameMeans, norm)

        music_mask_dirs = [os.path.join(wholeDir, folder, x) for x in os.listdir(os.path.join(wholeDir, folder)) if
                           'music_mask_' in x and 'txt' in x]

        makeVoicefor_mask(music_mask_dirs, True, fftangle)

        myloss = np.loadtxt(os.path.join(wholeDir, folder, 'myloss.txt'))
        m_loss = np.loadtxt(os.path.join(wholeDir, folder, 'm_loss.txt'))

        m_mean = np.loadtxt(os.path.join(wholeDir, folder, 'm_mean.txt'))
        m_var = np.loadtxt(os.path.join(wholeDir, folder, 'm_var.txt'))

        fig = plt.figure()
        host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.axis['right'].set_visible(False)
        par1.axis['right'].set_visible(True)
        par1.set_ylabel('m_entropy=∑log(m_ij))')
        par1.axis['right'].major_ticklabels.set_visible(True)
        par1.axis['right'].label.set_visible(True)
        fig.add_axes(host)
        host.set_xlabel('epoch')
        host.set_ylabel('|preOutput-TrueOutput|²')
        p1, = host.plot(np.arange(len(myloss)), myloss/(K*T), label='|preOutput-TrueOutput|²')
        p2, = par1.plot(np.arange(len(m_loss)), m_loss/(K*T), label='m_entropy=∑log(m_ij)); ')
        plt.title("two parts of loss normalized /KT ")
        host.legend()
        # 轴名称，刻度值的颜色
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        plt.savefig(os.path.join(wholeDir, folder, 'loss_two_parts.png'), dpi=150)
        plt.clf()

        fig = plt.figure()
        host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.axis['right'].set_visible(False)
        par1.axis['right'].set_visible(True)
        par1.set_ylabel('m_var')
        par1.axis['right'].major_ticklabels.set_visible(True)
        par1.axis['right'].label.set_visible(True)
        fig.add_axes(host)
        host.set_xlabel('epoch')
        host.set_ylabel('m_mean')
        p1, = host.plot(np.arange(len(m_mean)), m_mean, label='m_mean')
        p2, = par1.plot(np.arange(len(m_var)), m_var, label='m_var')
        plt.title("m_changes")
        host.legend()
        # 轴名称，刻度值的颜色
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        plt.savefig(os.path.join(wholeDir, folder, 'm_changes.png'), dpi=150)
        plt.clf()

        outputgrad = np.loadtxt(os.path.join(wholeDir, folder, 'outputGrad.txt'),comments=['o', '[', '#', 'l'])
        plt.figure()
        plt.imshow(outputgrad, cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar()
        plt.title('last output grad')
        plt.ylabel('31 tokens')
        plt.xlabel('%d time indices'%(outputgrad.shape[1]))
        plt.savefig(os.path.join(wholeDir, folder, 'outputGrad.png'), dpi=150)
        plt.clf()

        myloss_grad_mean = np.loadtxt(os.path.join(wholeDir, folder, 'myloss_grad_mean.txt'))
        mloss_grad_mean = np.loadtxt(os.path.join(wholeDir, folder, 'mloss_grad_mean.txt'))

        myloss_grad_var = np.loadtxt(os.path.join(wholeDir, folder, 'myloss_grad_var.txt'))
        mloss_grad_var = np.loadtxt(os.path.join(wholeDir, folder, 'mloss_grad_var.txt'))

        fig = plt.figure()
        host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.axis['right'].set_visible(False)
        par1.axis['right'].set_visible(True)
        par1.set_ylabel('second 1/m')
        par1.axis['right'].major_ticklabels.set_visible(True)
        par1.axis['right'].label.set_visible(True)
        fig.add_axes(host)
        host.set_xlabel('epoch')
        host.set_ylabel('first mGrad')
        p1, = host.plot(np.arange(len(myloss_grad_mean)), myloss_grad_mean, label='first mGrad')
        p2, = par1.plot(np.arange(len(mloss_grad_mean)), mloss_grad_mean, label='second 1/m')
        plt.title("grad's mean of two parts of loss")
        host.legend()
        # 轴名称，刻度值的颜色
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        plt.savefig(os.path.join(wholeDir, folder, 'loss_two_parts_grad_mean.png'), dpi=150)
        plt.clf()

        fig = plt.figure()
        host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
        par1 = ParasiteAxes(host, sharex=host)
        host.parasites.append(par1)
        host.axis['right'].set_visible(False)
        par1.axis['right'].set_visible(True)
        par1.set_ylabel('second 1/m')
        par1.axis['right'].major_ticklabels.set_visible(True)
        par1.axis['right'].label.set_visible(True)
        fig.add_axes(host)
        host.set_xlabel('epoch')
        host.set_ylabel('first mGrad')
        p1, = host.plot(np.arange(len(myloss_grad_var)), myloss_grad_var, label='first mGrad')
        p2, = par1.plot(np.arange(len(mloss_grad_var)), mloss_grad_var, label='second 1/m')
        plt.title("grad's var of two parts of loss")
        host.legend()
        # 轴名称，刻度值的颜色
        host.axis['left'].label.set_color(p1.get_color())
        par1.axis['right'].label.set_color(p2.get_color())
        plt.savefig(os.path.join(wholeDir, folder, 'loss_two_parts_grad_var.png'), dpi=150)
        plt.clf()

        fft_mean = np.loadtxt(os.path.join(wholeDir, folder, 'fft_mean.txt'))
        plt.plot(np.arange(len(fft_mean)), fft_mean)
        plt.title('fft\'s mean changes with epoch')
        plt.savefig(os.path.join(wholeDir, folder, 'fftmean_changes.png'), dpi=150)
        plt.clf()





