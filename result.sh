# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python result.py




#调试
#THEANO_FLAGS="floatX=float32,device=gpu,on_unused_input=warn,optimizer=fast_compile" python config.py



