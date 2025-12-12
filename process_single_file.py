from model import *
import timeit
import os
import warnings
warnings.filterwarnings('ignore')

root_path = "project/Safe Lane"
file_name = "DJI_0008_transformed_processed.csv"

start_timer = timeit.default_timer()
if __name__=="__main__":
    os.chdir(root_path)

    RunDataProcessingType2(
        raw_file_path=os.path.join(root_path,"raw",file_name),
        file_name=file_name.split(".")[0],
        mpp=0.0272767352,
        image_reference="DJI_0001_1110.png",
        id_regions=["Ciclovia","Faixa 1","Faixa 2","Faixa 3","Acesso 1","Acesso 2"],
        regions=[[(1, 447), (365, 471), (1090, 471), (1591, 471), (1918, 430), (1918, 484), (1785, 507), (1604, 527), (1095, 527), (1017, 527), (838, 531), (660, 531), (481, 528), (128, 512), (1, 502)],[(0, 502), (130, 514), (485, 528), (837, 530), (1018, 528), (1604, 530), (1918, 485), (1918, 595), (1810, 612), (1587, 635), (1131, 668), (905, 678), (691, 681), (461, 678), (255, 670), (52, 657), (1, 650), (0, 518)],[(0, 797), (237, 814), (475, 817), (980, 804), (1264, 781), (1621, 744), (1914, 708), (1918, 595), (1738, 620), (1511, 644), (1132, 667), (690, 680), (464, 677), (254, 668), (0, 651), (0, 778)],[(1, 932), (511, 947), (1120, 935), (1597, 878), (1907, 837), (1911, 707), (1588, 748), (1191, 788), (828, 810), (618, 817), (477, 818), (1, 795), (0, 920)],[(1122, 935), (1601, 877), (1532, 962), (1537, 1078), (1228, 1078), (1221, 1025), (1127, 942)],[(1091, 471), (1588, 471), (1488, 385), (1462, 0), (1162, 0), (1175, 375), (1104, 461)]],
        id_conflict_zones=["ZC-OM","ZA-B","ZA-OM","ZConf1"],
        conflict_zones=[[(1097, 470), (1100, 671), (1400, 652), (1392, 478)],[(1097, 470), (912, 474), (907, 528), (1098, 525)],[(1098, 534), (1101, 672), (1012, 677), (908, 680), (904, 530), (1098, 525)],[(1100, 471), (1095, 534), (1407, 531), (1408, 480)]],
        video_heigth=1080,
        video_width=1920,
        flip_h=False,
        flip_v=False,
        green_open_time=[0],
        force_processing=False)

stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")