def main():
    import torch,cv2, pdb, traceback,sys
    from time import sleep
    import os,sys,time
    from upcunet_v3 import RealWaifuUpScaler
    from inference_video import VideoRealWaifuUpScaler
    from time import time as ttime
    root_path=os.path.abspath('.')
    sys.path.append(root_path)
    from config import half, model_path2, model_path3, model_path4, tile, cache_mode,scale, input_dir, output_dir,device,inp_path,opt_path,mode,nt,n_gpu,encode_params,p_sleep,decode_sleep,alpha
    print("tile=%s\thalf=%s\tscale=%s"%(tile,half,scale))
    ####别动下面，乱调alpha会鬼畜
    if (alpha > 1.3 or alpha < 0.75):
        print("warning:alpha should be in range(0.75,1.3)")
        alpha = min(max(0.75, alpha), 1.3)
    try:
        print("torch version",torch.__version__)
        print("torch cuda version",torch.version.cuda)
        ngpu=torch.cuda.device_count()
        print("total GPU number",ngpu)
        for i in range(ngpu):
            print(torch.cuda.get_device_properties(i))
        if(torch.cuda.is_available()==False):
            print("Can't find nvidia drive, try using CPU to super resolve it.")
            device="cpu"
        if(mode=="image"):
            upscaler2x = RealWaifuUpScaler(scale, eval("model_path%s" % scale), half, device)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(root_path, "tmp"), exist_ok=True)
            for name in os.listdir(input_dir):
                tmp=name.split(".")
                inp_path=os.path.join(input_dir,name)
                suffix = tmp[-1]
                prefix=".".join(tmp[:-1])
                tmp_path = os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
                os.link(inp_path, tmp_path)
                frame = cv2.imread(tmp_path)[:,:,[2,1,0]]
                t0=ttime()
                result=upscaler2x(frame,tile,cache_mode=cache_mode,alpha=alpha)[:,:,::-1]
                t1=ttime()
                print(prefix,"done",t1-t0)
                tmp_opt_path=os.path.join(root_path, "tmp", "%s.%s" % (int(time.time() * 1000000), suffix))
                cv2.imwrite(tmp_opt_path, result)
                n=0
                while(1):
                    if(n==0):suffix="_%sx.png" % (scale)
                    else:suffix="_%sx_%s.png" % (scale,n)#
                    if(os.path.exists(os.path.join(output_dir, prefix + suffix))==False):break
                    else:n+=1
                final_opt_path=os.path.join(output_dir, prefix + suffix)
                os.rename(tmp_opt_path,final_opt_path )
                os.remove(tmp_path)
                # tcount=0
                # for i in range(20):
                #     t0 = ttime()
                #     result = upscaler2x(frame)
                #     t1 = ttime()
                #     tcount+=(t1-t0)
                # print (prefix,tcount)
        elif(mode=="video"):
            video_upscaler = VideoRealWaifuUpScaler(nt, n_gpu, scale, half, tile, p_sleep,decode_sleep,encode_params)
            tmp_path=video_upscaler(inp_path, opt_path)
            os.remove(tmp_path)
    except:
        traceback.print_exc()


    os.system('pause')
main()