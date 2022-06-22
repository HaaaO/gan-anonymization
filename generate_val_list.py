import os
import numpy as np
import cv2
from os import listdir, mkdir
from os.path import isfile, join, isdir
import dlib
from PIL import Image
import argparse
import csv

def get_lndm(path_img, path_out, start_id = 0, dlib_path=""):
    dir_proc = {'msk':'msk', 'org':'orig', 'clr':'clr', 'lnd':'lndm'}

    for dir_it in dir_proc:
        if os.path.isdir(path_out + dir_proc[dir_it]) == False:
            os.mkdir(path_out + dir_proc[dir_it])

    folder_list = [f for f in listdir(path_img)]
    folder_list.sort()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path+"shape_predictor_68_face_landmarks.dat")

    line_px = 1
    res_w = 178
    res_h = 218
    print(folder_list)
    folder_list = folder_list[1:]
    print(folder_list)
    for fld in folder_list[:]:
        imglist_all = [f[:-4] for f in listdir(join(path_img, fld)) if isfile(join(path_img, fld, f)) and f[-4:] == ".jpg"]
        imglist_all.sort(key=str)
        imglist_all = imglist_all[start_id:]

        for dir_it in dir_proc:
            if os.path.isdir(join(path_out, dir_proc[dir_it], fld)) == False:
                os.mkdir(join(path_out, dir_proc[dir_it], fld))

        land_mask = True
        crop_coord = []
        for it in range(len(imglist_all)):
            clr = cv2.imread(join(path_img, fld, imglist_all[it]+".jpg"), cv2.IMREAD_ANYCOLOR)
            # print(clr.shape)
            img = clr.copy()
            img_dlib = np.array(clr[:, :, :], dtype=np.uint8)
            dets = detector(img_dlib, 1)

            for k_it, d in enumerate(dets):
                if k_it != 0:
                    continue
                landmarks = predictor(img_dlib, d)
                # print(img_dlib)
                # print(landmarks.part(42))
                # centering
                c_x = int((landmarks.part(42).x + landmarks.part(39).x) / 2)
                c_y = int((landmarks.part(42).y + landmarks.part(39).y) / 2)
                w_r = int((landmarks.part(42).x - landmarks.part(39).x)*4)
                h_r = int((landmarks.part(42).x - landmarks.part(39).x)*5)
                w_r = int(h_r/res_h*res_w)

                w, h = int(w_r * 2), int(h_r * 2)
                pd = int(w) # padding size
                
                img_p = np.zeros((img.shape[0]+pd*2, img.shape[1]+pd*2, 3), np.uint8) * 255
                img_p[:, :, 0] = np.pad(img[:, :, 0], pd, 'edge')
                img_p[:, :, 1] = np.pad(img[:, :, 1], pd, 'edge')
                img_p[:, :, 2] = np.pad(img[:, :, 2], pd, 'edge')
                
                visual = img_p[c_y - h_r+pd:c_y + h_r+pd, c_x - w_r+pd:c_x + w_r+pd]

                crop_coord.append([c_y - h_r, c_y + h_r, c_x - w_r, c_x + w_r, pd, imglist_all[it]+".jpg"])
                t_x, t_y = int(c_x - w_r), int(c_y - h_r)

                ratio_w, ratio_h = res_w/w, res_h/h

                visual = cv2.resize(visual, dsize=(res_w, res_h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(join(path_out, dir_proc['clr'], fld, imglist_all[it]+".jpg"), visual) #saving crop
                cv2.imwrite(join(path_out, dir_proc['org'], fld, imglist_all[it]+".jpg"), clr) # saving original

                if land_mask:
                    img_lndm = np.ones((res_h, res_w, 3), np.uint8) * 255

                    def draw_line(offset, pt_st, pt_end):
                        cv2.line(img_lndm, (int((landmarks.part(offset + pt_st).x - t_x) * ratio_w), int((landmarks.part(offset + pt_st).y - t_y) * ratio_h)), (int((landmarks.part(offset + pt_end).x - t_x) * ratio_w), int((landmarks.part(offset + pt_end).y - t_y) * ratio_h)), (0, 0, 255), line_px)

                    for i in range(16):
                        draw_line(0, i, i+1)

                    for i in range(3):
                        draw_line(27, i, i+1)

                    for i in range(4):
                        draw_line(60, i, i+1)

                    for i in range(3):
                        draw_line(64, i, i+1)

                    draw_line(0, 67, 60)

                    result = Image.fromarray((img_lndm).astype(np.uint8))
                    result.save(join(path_out, dir_proc['lnd'], fld, imglist_all[it]+".jpg"))

                    img_msk = np.ones((res_h, res_w, 3), np.uint8) * 255

                    contours = np.zeros((0, 2))
                    contours = np.concatenate((contours, np.array([[(landmarks.part(0).x - t_x) * ratio_w, (landmarks.part(19).y - t_y) * ratio_h]])), axis=0)
                    for p in range(17):
                        contours = np.concatenate((contours, np.array([[(landmarks.part(p).x - t_x) * ratio_w, (landmarks.part(p).y - t_y) * ratio_h]])),axis=0)
                    contours = np.concatenate((contours, np.array([[(landmarks.part(16).x - t_x) * ratio_w, (landmarks.part(24).y - t_y) * ratio_h]])),axis=0)
                    contours = contours.astype(int)
                    cv2.fillPoly(img_msk, pts=[contours], color=(0, 0, 0))
                    result = Image.fromarray((img_msk).astype(np.uint8))
                    result.save(join(path_out, dir_proc['msk'], fld, imglist_all[it]+".jpg"))

        #np.save(join(path_out, dir_proc['org'], fld, 'crop_coord.npy'), crop_coord) #crop coordinates
        print("folder done",fld)

def extract_list(path_in, path_out):

    folder_list = [f for f in listdir(path_in)]
    folder_list.sort()

    counter = 1

    with open(join(args.output, "processed_file.txt"), 'w') as output:


        for fld in folder_list:
            imglist_all = [join(fld, f[:]) for f in listdir(join(path_in, fld)) if isfile(join(path_in, fld, f)) and f[-4:] == ".jpg"]
            imglist_all.sort()
            for img in imglist_all:
                print(counter, img)
                output.write(str(counter) + ";" + img + '\n')

                counter += 1

def transfer_to_number(path_in, path_out,list_path):
    dir_proc = ['msk', 'clr', 'lndm']

    all_images_list = []

    with open(list_path,'r') as f:
        for line in f:
            all_images_list.append(line.strip().split(";"))

    print(all_images_list[0])

    for f in dir_proc:
        for i in all_images_list:
            img_curr_path = path_in + f + "/" + i[1]
            img_out_path = path_out + f + "/0/" +str(i[0]).zfill(6) + ".jpg"


            # img_curr_path = path_in + f  + "10001312@N04/landmark_aligned_face.616.8530792823_f2ce4a8110_o.jpg"
            # img_out_path = path_out + f + "/0/" + str(43).zfill(6) + ".jpg"
            # print(img_curr_path)
            # print(img_out_path)
            clr = cv2.imread(img_curr_path, cv2.IMREAD_ANYCOLOR)
            # print(clr)
            cv2.imwrite(img_out_path, clr)
     

            # clr = cv2.imread(img_curr_path, cv2.IMREAD_ANYCOLOR)
            # cv2.imwrite(join(), clr) # saving original


def compare(path_in):

    success_list = []
    with open(path_in + "/processed_file.txt",'r') as f:
        for line in f:
            success_list.append(line.strip().split(";")[1].split("/")[1])

    print(success_list)
    print(len(success_list))

    
    if "landmark_aligned_face.2174.9524511337_2d3c153c15_o.jpg" in success_list:
        print("success")

    total = 0
    counter = 0

    with open(path_in + "/adience_frontal_landmark_val.txt", 'r') as f:

        for line in f:

            total += 1
            file_name = line.strip().split(";")[0].split("/")[2]

            if file_name in success_list:
                counter += 1


    print("percentage:", counter / total)


def generate_val_list(path_in):

    success_list = []
    success_dic = {}
    with open(path_in + "processed_file.txt",'r') as f:
        for line in f:
            success_list.append("aligned/" + line.strip().split(";")[1])
            success_dic["aligned/" + line.strip().split(";")[1]] = line.strip().split(";")[0]

    with open("final_valid_list.txt", 'w') as output:       
        with open(path_in + "adience_frontal_landmark_val.txt", 'r') as f:

            for line in f:
                first_part = line.strip().split(";")[0]
                second_part = line.strip().split(";")[1:]
                print(first_part)
                print(second_part)



                if first_part in success_list:


                    output.write(str((int(success_dic[first_part])-1)).zfill(6) + ".jpg" + ";" + second_part[0] + ";" + second_part[1] + ";" + second_part[2]+"\n")



    # print(success_list)
    # print(success_dic)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='directory with input data', default='')
    args = parser.parse_args()
    

    generate_val_list(args.input)



    