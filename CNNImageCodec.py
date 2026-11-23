#This code is the simplest example of image compression based on neural networks
#Comparison with JPEG is provided as well
#It is a demonstation for Information Theory course
#Written by Evgeny Belyaev, August 2026.
import sys
import os
import math
from dataclasses import dataclass
import numpy
from matplotlib import pyplot as plt
from PIL import Image
import imghdr
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
#from keras import backend as K
import tensorflow.keras.backend as K
from skimage.metrics import structural_similarity as ssim
import numpy as np
from RangeCoder import ArithmeticEncoder, ArithmeticDecoder
from bj_delta import *
from tensorflow.keras.callbacks import EarlyStopping

# Restrict GPU memory use
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#source folder with test images
testfolder = './test/'
#source folder with train images
trainfolder = './train/'
#size of test and train images
w=128
h=128
#If 0, then the training will be started, otherwise the model will be readed from a file
LoadModel = 1
#Number of bits for representation of the layers sample in the training process
bt = 2
#Training parameters 
epochs = 2000
#Model parameters
batch_sizeM1 = 24
n1M1=128
n2M1=32
n3M1=16

#Number of images to be compressed and shown from the test folder
NumImagesToCompress = 20
#Number of images to be shown from the test folder
NumImagesToShow = 5

#Number of bits for representation of the layers sample
b = 2

#Range coder variables
ALPHABET_SIZE = pow(2, b)   
CONTEX_SIZE = ALPHABET_SIZE*n3M1 
freqs = np.ones((CONTEX_SIZE, ALPHABET_SIZE+1), dtype=np.int64)
cumfreqs = np.zeros(ALPHABET_SIZE+1,  dtype=np.int64)
MAX_FREQ_SUM=4096

#Compute PSNR in RGB domain
def PSNR_RGB(image1,image2):
    width, height = image1.size
    I1 = numpy.array(image1.getdata()).reshape(image1.size[0], image1.size[1], 3)
    I2 = numpy.array(image2.getdata()).reshape(image2.size[0], image2.size[1], 3)
    I1 = numpy.reshape(I1, width * height * 3)
    I2 = numpy.reshape(I2, width * height * 3)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    mse = numpy.mean((I1 - I2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        psnr=100.0
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    #print("PSNR = %5.2f dB" % psnr)
    return psnr

#Compute PSNR between two vectors
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

#reads all images from folder and puts them into x array
def LoadImagesFromFolder (foldername):
    dir_list = os.listdir(foldername)
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('')
        else:
            Nmax = Nmax + 1

    x = numpy.zeros([Nmax, w, h, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        filetype = imghdr.what(fullname)
        if filetype is None:
            print('Unknown image format for file: ', name)
        else:
            print('\rProgress: N = %i' % N, end='')
            image = Image.open(fullname)
            I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
            x[N, :, :, :] = I1
            N = N + 1
    print()
    return x

def residual_block(input_tensor, filters, kernel_size=(3, 3), strides=1, downsample=False):
    #Residual block with optional downsampling and channel adjustment
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust skip connection if needed
    if downsample or input_tensor.shape[-1] != filters:
        skip = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(input_tensor)
        skip = layers.BatchNormalization()(skip)
    else:
        skip = input_tensor

    x = layers.Add()([x, skip])
    return layers.Activation('relu')(x)

def ImageCodecModel(trainfolder, trainwithnoise,btrain):
    input = layers.Input(shape=(w, h, 3))
    #Encoder
    x = layers.Conv2D(n1M1, (7, 7), padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, n1M1, kernel_size=(5, 5))
    x = layers.AveragePooling2D((2, 2), padding='same')(x)
    x = residual_block(x, n2M1, kernel_size=(5, 5))
    x = layers.AveragePooling2D((2, 2), padding='same')(x)

    x = residual_block(x, n3M1, kernel_size=(3, 3))
    encoder_output = layers.AveragePooling2D((2, 2), padding='same')(x)

    # Add noise during training to simulate quantization
    if trainwithnoise == 1:
        max_val = tensorflow.reduce_max(encoder_output)
        noise = tensorflow.random.uniform(
            shape=tensorflow.shape(encoder_output),
            minval=-1.0 / (2 ** (btrain + 1)),
            maxval=1.0 / (2 ** (btrain + 1)),
            dtype=tensorflow.float32
        )
        decoder_input = encoder_output + max_val * noise
    else:
        decoder_input = encoder_output

    # Decoder
    x = layers.UpSampling2D(size=(2, 2))(decoder_input)
    x = layers.Conv2D(n3M1, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(n2M1, (5, 5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(n1M1, (7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input, encoder_output, name='encoder')
    decoder = Model(decoder_input, output, name='decoder')
    autoencoder = Model(input, output, name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.summary()

    encname = './models/encoder_%i_%i.h5' % (trainwithnoise,btrain)
    decname = './models/decoder_%i_%i.h5' % (trainwithnoise,btrain)

    if LoadModel == 0:
        print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
        xtrain = LoadImagesFromFolder(trainfolder)
        xtrain = xtrain / 255
        with tensorflow.device('gpu'):    
            history = autoencoder.fit(
                xtrain, xtrain,
                epochs=epochs,
                batch_size=batch_sizeM1,
                shuffle=True
            )
        encoder.save(encname)
        decoder.save(decname)
    else:
        encoder = keras.models.load_model(encname)
        decoder = keras.models.load_model(decname)
            
    return encoder,decoder

def initfreqs():
    freqs[:, :ALPHABET_SIZE] = 1
    freqs[:, ALPHABET_SIZE] = ALPHABET_SIZE
    
def encode_symbol(enc, symbol, history):
    freq_dict = freqs[history]
    #print(symbol)
    #print(freq_dict)
    enc.write(freq_dict, symbol, ALPHABET_SIZE)
    freqs[history][symbol] = freqs[history][symbol]+1
    freqs[history][ALPHABET_SIZE] = freqs[history][ALPHABET_SIZE]+1
    #if freqs[history].sum() > MAX_FREQ_SUM:
    #    freqs[history] = np.maximum(freqs[history] // 2, 1)

def decode_symbol(dec, history):
    freq_dict = freqs[history]
    cum = 0
    for i in range(ALPHABET_SIZE):
        cumfreqs[i] = cum
        cum += freq_dict[i]
    cumfreqs[ALPHABET_SIZE] = cum
    symbol =  dec.read(freq_dict, cumfreqs, ALPHABET_SIZE)
    #print(symbol)
    freqs[history][symbol] = freqs[history][symbol]+1
    #if freqs[history].sum() > MAX_FREQ_SUM:
    #    freqs[history] = np.maximum(freqs[history] // 2, 1) 
    return symbol

#Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
@dataclass(frozen=True)
class EntropyContextMap:
    spatial: tuple
    guides: tuple


def EntropyContext(layers, row, col, channel, previous, mode):
    channels = layers.shape[-1]
    if isinstance(mode, EntropyContextMap):
        direction = mode.spatial[channel]
        left = int(layers[row, col - 1, channel]) if col else 0
        above = int(layers[row - 1, col, channel]) if row else 0
        if not col:
            neighbor = above
        elif not row or direction == 'left':
            neighbor = left
        elif direction == 'above':
            neighbor = above
        else:
            neighbor = (left + above + 1) // 2
        guide_channel = mode.guides[channel]
        guide = previous if guide_channel == -1 else (
            int(layers[row, col, guide_channel]) if guide_channel >= 0 else 0)
        return channels * (neighbor + ALPHABET_SIZE * guide) + channel
    if mode == 'previous':
        return channels * previous + channel
    raise ValueError(f'Unknown context mode: {mode}')


def ResetEntropyFrequencies(prior, mode, channels):
    global freqs
    contexts = EntropyContextCount(mode, channels)
    shape = (contexts, ALPHABET_SIZE + 1)
    if freqs.shape != shape:
        freqs = np.empty(shape, dtype=np.int64)
    initfreqs()
    if prior is not None:
        freqs[:, :ALPHABET_SIZE] = prior
        freqs[:, ALPHABET_SIZE] = prior.sum(axis=1)


def EntropyContextCount(mode, channels):
    return channels * ALPHABET_SIZE ** (2 if isinstance(mode, EntropyContextMap) else 1)


def EntropyEncoder (filename,enclayers,size_z,size_h,size_w,mode='previous',prior=None):
    ResetEntropyFrequencies(prior, mode, size_w)
    with open(filename, "wb") as out:
        enc = ArithmeticEncoder(16, out)
        # Инициализация истории последних MAX_ORDER символов (заполняем нулями)
        history = 0
        for z in range(size_z):
            for h in range(size_h):
                for w in range(size_w):
                    symbol = int(enclayers[z][h][w])
                    context = EntropyContext(enclayers, z, h, w, history, mode)
                    encode_symbol(enc, symbol, context)
                    # Обновляем историю: удаляем первый элемент и добавляем новый символ
                    history = symbol
        enc.finish()

#Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyDecoder (filename,size_z,size_h,size_w,mode='previous',prior=None):
    ResetEntropyFrequencies(prior, mode, size_w)
    declayers = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    with open(filename, "rb") as inp:
        dec = ArithmeticDecoder(16, inp)
        history = 0
        for z in range(size_z):
            for h in range(size_h):
                for w in range(size_w):
                    context = EntropyContext(declayers, z, h, w, history, mode)
                    symbol = decode_symbol(dec, context)
                    declayers[z][h][w]=symbol
                    history = symbol
    return declayers

#This function is searching for the JPEG quality factor (QF)
#which provides neares compression to TargetBPP
def JPEGRDSingleImage(X,TargetBPP,i):
    X = X*255
    image = Image.fromarray(X.astype('uint8'), 'RGB')
    width, height = image. size
    realbpp = 0
    realQ = 0
    for Q in range(101):
        image.save('test.jpeg', "JPEG", quality=Q)
        bytesize = os.path.getsize('test.jpeg')
        bpp = bytesize*8/(width*height)
        if Q == 0 or abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realQ = Q
    image.save('test.jpeg', "JPEG", quality=realQ)
    image_dec = Image.open('test.jpeg')
    I1 = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    I2 = numpy.array(image_dec.getdata()).reshape(image_dec.size[0], image_dec.size[1], 3)

    #print('\n\n Size = ',numpy.shape(I1))
    psnr = ssim(I1[ :, :, 0], I2[ :, :, 0],data_range=255.0)
    psnr = psnr + ssim(I1[ :, :, 1], I2[ :, :, 1],data_range=255.0)
    psnr = psnr + ssim(I1[ :, :, 2], I2[ :, :, 2],data_range=255.0)
    realpsnr=psnr/3.0
    JPEGfilename = './jpg/image%i.jpeg' % i
    image.save(JPEGfilename, "JPEG", quality=realQ)
    return realQ, realbpp, realpsnr

def QuantizeLatents(encoded_layers, bits):
    maxima = encoded_layers.max(axis=(1, 2, 3)).astype(numpy.float16)
    scales = maxima[:, None, None, None]
    normalized = numpy.divide(encoded_layers, scales, out=numpy.zeros_like(encoded_layers),
                              where=scales != 0)
    return (numpy.clip(normalized, 0, 0.9999999) * (2 ** bits)).astype(numpy.int32), maxima


def DequantizeLatents(symbols, maxima, bits):
    return ((symbols.astype(numpy.float32) + 0.5) / (2 ** bits)
            * maxima[:, None, None, None])


def NeuralCompressor(enc,dec,btest,mode='previous',prior=None):
    encoded_layers = enc.predict(xtest, batch_size=NumImagesToCompress)[:NumImagesToCompress]
    encoded_layers1, max_encoded_layers = QuantizeLatents(encoded_layers, btest)

    #Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(NumImagesToCompress, numpy.float16, 'C')
    declayers = numpy.zeros((NumImagesToCompress,n3M1, n3M1, n3M1), numpy.uint8, 'C')
    for i in range(NumImagesToCompress):
        binfilename = './bin/image%i.bin' % i
        EntropyEncoder(binfilename, encoded_layers1[i], n3M1, n3M1, n3M1, mode, prior)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = EntropyDecoder(binfilename, n3M1, n3M1, n3M1, mode, prior)

    #Dequantization and denormalization of each layer
    #print(bpp)
    encoded_layers_quantized = DequantizeLatents(declayers, max_encoded_layers, btest)
    decoded_imgsQ = dec.predict(encoded_layers_quantized, batch_size=NumImagesToCompress)
    return bpp, decoded_imgsQ
   

# Main function
if __name__ == '__main__':
    #Load test images
    xtest = LoadImagesFromFolder(testfolder)
    xtest = xtest / 255

    #btrainarray = np.array([1, 2, 3, 4])
    btrainarray = np.array([2])
    LoadModel = 1
    if LoadModel==0:
    #Обучение всех моделей    
        #Train/load the model
        for i in range(len(btrainarray)):
            encoder, decoder = ImageCodecModel(trainfolder,1,btrainarray[i])
    LoadModel = 1

    QJPEG   = np.zeros(4)
    QAE     = np.zeros(4)
    bppJPEG = np.zeros(4)
    bppAE   = np.zeros(4)
    k = 0
    btrain=2
    for btrain in range(1,5):
        btest = btrain
        #Range coder variables
        ALPHABET_SIZE = pow(2, btest)   
        CONTEX_SIZE = ALPHABET_SIZE*n3M1 
        freqs = np.ones((CONTEX_SIZE, ALPHABET_SIZE+1), dtype=np.int64)
        cumfreqs = np.zeros(ALPHABET_SIZE+1,  dtype=np.int64)

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        encoder, decoder = ImageCodecModel(trainfolder,1,btrain)
        sys.stdout = original_stdout
        bpp, decoded_imgsQ = NeuralCompressor(encoder,decoder,btest)
        
        print('btrain = %i, btest = %i,' % (btrain,btest))
        print('#i\tbpp\tQ\tJPEGbpp\tJPEGQ')
        BPPAve = 0;
        QAve = 0;
        BPPAveJPEG = 0;
        QAveJPEG = 0;
        for i in range(NumImagesToCompress):
            ssim2=0;
            ssim2 = ssim(xtest[i, :, :, 0], decoded_imgsQ[i, :, :, 0], data_range=1.0)
            ssim2 = ssim2 + ssim(xtest[i, :, :, 1], decoded_imgsQ[i, :, :, 1], data_range=1.0)
            ssim2 = ssim2 + ssim(xtest[i, :, :, 2], decoded_imgsQ[i, :, :, 2], data_range=1.0)
            ssim2 = ssim2 / 3.0
            JPEGQP, JPEGrealbpp, JPEGrealssim = JPEGRDSingleImage(xtest[i, :, :, :], bpp[i], i)
            BPPAve = BPPAve+bpp[i];
            QAve = QAve + ssim2;
            BPPAveJPEG = BPPAveJPEG+JPEGrealbpp;
            QAveJPEG = QAveJPEG + JPEGrealssim;
            print("#%i\t%.3f\t%.3f\t%.3f\t%.3f" % (i,bpp[i],ssim2,JPEGrealbpp,JPEGrealssim))
        print("Average: %.3f\t%.3f\t%.3f\t%.3f\n" % (BPPAve/NumImagesToCompress,QAve/NumImagesToCompress,BPPAveJPEG/NumImagesToCompress,QAveJPEG/NumImagesToCompress))
        QJPEG[k]   = QAveJPEG/NumImagesToCompress
        QAE[k]     = QAve/NumImagesToCompress
        bppJPEG[k] = BPPAveJPEG/NumImagesToCompress
        bppAE[k]   = BPPAve/NumImagesToCompress
        k = k + 1

    plt.figure(figsize=(10, 6))
    plt.plot(bppJPEG, QJPEG, label="JPEG", color="blue", linewidth=2, marker="o")
    plt.plot(bppAE, QAE, label="AE", color="red", linewidth=2, marker="o")
    plt.xlabel("bpp", fontsize=12)
    plt.ylabel("SSIM", fontsize=12)
    #plt.title("Сравнение зависимости SSIM от bpp для JPEG и AE", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7) # Включаем сетку
    plt.legend(fontsize=11)
    plt.savefig("rd-performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    BDRate = bj_delta(bppJPEG, QJPEG, bppAE, QAE, mode=0)
    print("BD-Rate = %.3f" % (BDRate))
    BDRate = bj_delta(bppJPEG, QJPEG, bppAE, QAE, mode=1)
    print("BD-Rate = %.3f" % (BDRate))
   

    VusualComparison = 1
    if VusualComparison==1:
        #Shows NumImagesToShow images from the test set
        #For each image the following results are presented
        #Original image (RAW)
        #Image, represented by the model (with noise additing during training)
        #Corresponding JPEG image at the same compression level
        #Q is the quality metric measured as SSIM
        #bpp is bit per pixel after compression (bpp for RAW data is 24 bpp)
        btest = 2
        LoadModel = 1
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        encoder, decoder = ImageCodecModel(trainfolder,1,btest)
        sys.stdout = original_stdout
        bpp, decoded_imgsQ = NeuralCompressor(encoder,decoder,btest)
        for i in range(NumImagesToShow):
                title=''
                plt.subplot(3, NumImagesToShow, i + 1).set_title(title, fontsize=10)
                if i==0:
                    plt.subplot(3, NumImagesToShow, i + 1).text(-50, 64, 'RAW')
                plt.imshow(xtest[i, :, :, :], interpolation='nearest')
                plt.axis(False)
        for i in range(NumImagesToShow):
            psnr = ssim(xtest[i, :, :, 0], decoded_imgsQ[i, :, :, 0],data_range=1.0)
            psnr = psnr + ssim(xtest[i, :, :, 1], decoded_imgsQ[i, :, :, 1],data_range=1.0)
            psnr = psnr + ssim(xtest[i, :, :, 2], decoded_imgsQ[i, :, :, 2],data_range=1.0)
            psnr=psnr/3.0
            title = 'Q=%2.2f bpp=%2.2f' % (psnr, bpp[i])
            plt.subplot(3, NumImagesToShow,NumImagesToShow + i + 1).set_title(title, fontsize=10)
            if i==0:
                plt.subplot(3, NumImagesToShow, NumImagesToShow + i + 1).text(-50, 64, 'AE')
            plt.imshow(decoded_imgsQ[i, :, :, :], interpolation='nearest')
            plt.axis(False)
        for i in range(NumImagesToShow):
            JPEGQP,JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(xtest[i, :, :, :], bpp[i],i)
            JPEGfilename = './jpg/image%i.jpeg' % i
            JPEGimage = Image.open(JPEGfilename)
            title = 'Q=%2.2f bpp=%2.2f' % (JPEGrealpsnr,JPEGrealbpp)
            plt.subplot(3, NumImagesToShow, 2*NumImagesToShow + i + 1).set_title(title, fontsize=10)
            if i==0:
                plt.subplot(3, NumImagesToShow, 2*NumImagesToShow + i + 1).text(-50, 64, 'JPEG')
            plt.imshow(JPEGimage, interpolation='nearest')
            plt.axis(False)
        plt.show()
