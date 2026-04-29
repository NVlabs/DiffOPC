#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from PIL import Image, ImageDraw


# In[3]:


#文件的读取
f= open("cells",encoding='utf-8')
contents=f.readlines()
f.close()


# In[14]:


#一行是一张图，每行用；分割polygon,每个;存储一个polygon的轮廓坐标
#将一行的str读为list
def image_str2image_list(image_str):
    image_list=[]
    for i in range(0,len(image_str)):
        image_list.append(image_str[i].split(";"))
    return image_list


# In[5]:


image_list=image_str2image_list(contents)



# In[13]:


#将一个区域字符串转为坐标列表

def Laout_ord_list_all(layou_list):
        laout_list=[]
        for i in range(0,len(layou_list)-1):
                #将第image个图片的第i个区域的字符串转换为坐标列表
            list=polygon_str2ord_list(layou_list[i])
            laout_list.append(list)
        return laout_list

def polygon_str2ord_list(polygon_str):
    #一个区域包含num_ord个顶点，创建容量为num_ord的列表
    num_ord=0
    for i in range(0,len(polygon_str)):
        if(polygon_str[i]==","):
            num_ord+=1
    #print("有%d个顶点"%num_ord)
    polygon_list=[[] for _ in range(num_ord)]
    #polygon_list[j]存放坐标，如['1', '2']即为（1，2）
    #j为当前顶点
    j=0    
    for i in range(0,len(polygon_str)):
        if(j<num_ord):
            #遇到"，"即为下一个坐标
            if(polygon_str[i]==','):
                j=j+1
            else:
                polygon_list[j]+=polygon_str[i]
    #去除列表中的空格，并将其读取为数字
    result_list=[]
    for i in range(0,len(polygon_list)):
        #将列表读取为字符串，按空格分隔
        space_separated_string =''.join(polygon_list[i])
        #将字符串读取为列表，按空格分隔
        word_list = space_separated_string.split()
        assert len(word_list)==2
        assert word_list[0]!=','
        assert word_list[1]!=','
        result_list.append(word_list)

    return result_list


# In[12]:


#将一个区域的坐标序列化
def ord_list2order_xulie(ord_list):
    ord_xulie=[]
    
    start_flag="sop"

    end_flag="eop"

    ord_xulie.append(start_flag)
    begin_ord=ord_list[0]
    ord_xulie.append(begin_ord[0])
    ord_xulie.append(begin_ord[1])

    # print(ord_xulie)
    #有len个顶点的图形要求len次步长
    for now_loc in range(0,len(ord_list)-1):
        next_loc=now_loc+1
        now_x=int(ord_list[now_loc][0])
        now_y=int(ord_list[now_loc][1])
        next_x=int(ord_list[next_loc][0])
        next_y=int(ord_list[next_loc][1])
        direction=""
        if(now_x==next_x):
            step=abs(now_y-next_y)
            if(now_y-next_y<0):
                direction="r"
            else:
                direction="l"
        if(now_y==next_y):
            step=abs(now_x-next_x)
            if(now_x-next_x>0):
                direction="u"
            else:
                direction="d"
        ord_xulie.append(direction)
        ord_xulie.append(str(step))
    ord_xulie.append(end_flag)
    return ord_xulie
# 将一张图的序列化


# In[11]:


def list2xulie(image_list):
    all_xulie=[]
    for image in range(0,len(image_list)):
        polygon_list=[]#存储每张图的polygon
        start_flag="sob"
        end_flag="eob"
        polygon_list.append(start_flag)
        #图片image_list[image]中有len()-1个区域（因为最后有个换行符），将每个区域序列化然后存进polygon_list中
        #不存储3个区域以下的图片
        if(len(image_list[image])-1>=3):
            for i in range(0,len(image_list[image])-1):
                #将第image个图片的第i个区域的字符串转换为坐标列表
                list=polygon_str2ord_list(image_list[image][i])
                #将第image个图片的第i个区域的坐标列表序列化，存为列表
                xulie_list=ord_list2order_xulie(list)
                        

                #xulie_str=','.join(xulie_list)
                for item in xulie_list:
                    polygon_list.append(item)

            polygon_list.append(end_flag)
            #print(polygon_list)
            #['sob', 'sop', '0', '0', 'd', '1014', 'r', '2048', 'u', '1014', 'l', '2048', 'eop', 'sop', '1', '1', 'd', '1034', 'r', '1556', 'u', '1034', 'l', '1556', 'eop', 'sop', '1', '1', 'd', '1034', 'r', '396', 'u', '1034', 'l', '396', 'eop', 'eob']
            print("finish image:%d"%image)
            all_xulie.append(polygon_list)
    return all_xulie


# In[2]:


def ord2image_show(polygons):
    #polygons = [[['1014', '0'], ['2048', '0'], ['2048', '148'], ['1014', '148'], ['1014', '0']], [['1014', '244'], ['2048', '244'], ['2048', '441'], ['2037', '441'], ['2037', '788'], ['1014', '788'], ['1014', '244']], [['1014', '884'], ['2037', '884'], ['2037', '1231'], ['2048', '1231'], ['2048', '1428'], ['1014', '1428'], ['1014', '884']], [['1014', '1524'], ['2048', '1524'], ['2048', '1721'], ['2037', '1721'], ['2037', '2048'], ['1014', '2048'], ['1014', '1524']]]

    img = Image.new('L', (2048, 2048), 0)
    draw = ImageDraw.Draw(img)


    # List of polygons

    # Convert string coordinates to integer tuples and draw each polygon
    for polygon in polygons:
        int_polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(int_polygon, fill=255)
    # img.save(path)
    img.show()
def ord2image_save(polygons,path):
    #polygons = [[['1014', '0'], ['2048', '0'], ['2048', '148'], ['1014', '148'], ['1014', '0']], [['1014', '244'], ['2048', '244'], ['2048', '441'], ['2037', '441'], ['2037', '788'], ['1014', '788'], ['1014', '244']], [['1014', '884'], ['2037', '884'], ['2037', '1231'], ['2048', '1231'], ['2048', '1428'], ['1014', '1428'], ['1014', '884']], [['1014', '1524'], ['2048', '1524'], ['2048', '1721'], ['2037', '1721'], ['2037', '2048'], ['1014', '2048'], ['1014', '1524']]]

    img = Image.new('L', (2048, 2048), 0)
    draw = ImageDraw.Draw(img)


    # List of polygons

    # Convert string coordinates to integer tuples and draw each polygon
    for polygon in polygons:
        int_polygon = [(int(x), int(y)) for x, y in polygon]
        if len(int_polygon)==1:
             continue
        else:
           draw.polygon(int_polygon, fill=255)
    img.save(path)
    # img.show()


# In[29]:


all_xulie=list2xulie(image_list)


# In[ ]:


#生成所有序列
import random

all_xulie=list2xulie(image_list)
# for i in range(0,len(all_xulie)):
random.shuffle(all_xulie)

max_len=max( len(xulie) for xulie in all_xulie)
print(max_len)

train=all_xulie[:int(0.8*len(all_xulie))]
val=all_xulie[int(0.8*len(all_xulie)):]
for i in range(0,len(train)):
    lists = train[i]
   #  lists+=['padding']*(max_len - len(lists))
    with open('/home/user/LML/self_define/dataset/train_file.txt', 'a') as file:
       file.write(' '.join(lists)+'\n')
      
    print("已写完%d"%i)
for i in range(0,len(val)):
    lists = all_xulie[i]
   #  lists+=['padding']*(max_len - len(lists))
    with open('/home/user/LML/self_define/dataset/valid_file.txt', 'a') as file:
       file.write(' '.join(lists)+'\n')
      
    print("已写完%d"%i)



