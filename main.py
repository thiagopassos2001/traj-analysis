 # Libs principais
from model import YoloMicroscopicDataProcessing
import geopandas as gpd
import pandas as pd
import shapely 

# Controle de execução e pastas
import timeit
import os

# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')

# Fluxo de execução para trabalhar com múltiplos arquivos
# Copiar o padrão de alterar
class Main():
    """
    Exemplo de uso, onde:
    * root_path é a pasta raiz do projeto
    * output_folder é a pasta dentro da raiz onde serão salvos os arquivos
    * prefix é o prefixo que os arquivos receberão
    * func é a função personalizada que será executada considerando como input o
      arquivo JSON padrão em "root_path" + "data/json"

    if __name__=="__main__":
        root_path = "C:/Users/User/Desktop/traj-analysis"
        output_folder = "data/hd1"
        run = Main()
        run.WorkflowPattern(
            root_path=root_path,
            output_folder=output_folder,
            prefix="Hd1_",
            func=RunHd1Analysis)
    """
    def __init__(self):
        print("Executando...")

    def WorkflowPattern(self,root_path,output_folder,prefix,func):
        """
        Função para executar a função "func" com input de um arquivo só
        Tem como input o JSON, com caminho padrão root_path + "data/json"
        Tem com output arquivos com o prefixo "prefix" na pasta "output_folder" dentro de "root_path"

        Exibe logs por padrão e caso dê erro em 1 arquivo, 
        exibe a mensagem sem comprometer toda a execução

        Não retorna nada
        """
        # Contador de tempo
        start_timer = timeit.default_timer()
        
        # Vai para o diretório indicado
        os.chdir(root_path)
        # Caminho dos json
        json_path = "data/json/"
        # Verifica se a pasta de destino existe, se não, cria
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        
        # Puxa todos os arquivos do diretório de inout
        all_files = os.listdir(json_path)
        
        # Execução em loop
        for f in all_files:
            print(f"Processando: {f}")
            try:
                result = func(os.path.join(json_path,f))
                result.to_csv(os.path.join(output_folder,prefix+f.replace(".json",".csv")),index=False)
                print(f"\tArquivo salvo com sucesso!")  
            except Exception as e:
                print(f"\tArquivo não salvo!")
                print(f"Erro na execução do arquivo {f}: {e}")
            finally:
                stop_timer = timeit.default_timer()
                count_timer = stop_timer - start_timer
                print(f"\tExecução: {int(count_timer//60)}min:{int(count_timer%60)}s")

def UpdateFormatToJSON(root_path):
    # Contador de tempo
    start_timer = timeit.default_timer()
    # Vai para o diretório indicado
    os.chdir(root_path)
    # Puxa todos os arquivos do diretório
    all_files = os.listdir("data/processed")
    for f in all_files:
        try:
            processed_file = f"data/processed/{f}"
            raw_file = f"data/raw/{os.path.basename(processed_file).split('Tratado_')[-1]}"
            parameter_file = f"data/parameter/{os.path.basename(processed_file).replace('Tratado_','Parametros_')}"
            json_file = f"data/json/{os.path.basename(processed_file).split('Tratado_')[-1].split('_transformed_rastreio')[0].replace('.csv','')}.json"

            # Verificação da existência dos arquivos
            if not os.path.isfile(processed_file):
                raise ValueError(f"{os.path.basename(processed_file)} não existe!")
            # if not os.path.isfile(raw_file):
            #     raise ValueError(f"{os.path.basename(raw_file)} não existe!")
            if not os.path.isfile(parameter_file):
                raise ValueError(f"{os.path.basename(parameter_file)} não existe!")
            if os.path.isfile(json_file):
                raise ValueError(f"{os.path.basename(json_file)} existe, execução encerrada!")
            
            model = YoloMicroscopicDataProcessing()
            model.raw_file = raw_file
            model.processed_file = processed_file
            model.parameter_file = parameter_file
            model.ImportParameter(model.parameter_file)
            model.CreateJSON(json_file)
            model.ImportFromJSON(json_file)

            print(f"{os.path.basename(json_file)} salvo!")
        except Exception as e:
            print(e)
        
        finally:
            stop_timer = timeit.default_timer()
            count_timer = stop_timer - start_timer
            print(f"Execução: {int(count_timer//60)}min:{int(count_timer%60)}s")

def RunHd1Analysis(file_path):
    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(file_path)

    result = []
    for t in model.green_open_time:
        try:
            result_ = model.Hd1Analysis(t)
            if not result_.empty:
                result.append(result_)
        except Exception as e:
            print(f"Erro {e} no {os.path.basename(file_path)} em {int(t//60)}:{int(t%60)}")

    result = pd.concat(result,ignore_index=True)
    result.insert(0,"file",os.path.basename(file_path))

    return result

def RunHd4Analysis(file_path):
    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(file_path)

    result = []
    for t in model.green_open_time:
        # try:
        result_ = model.Hd4Analysis(t)
        if not result_.empty:
            result.append(result_)
        # except Exception as e:
        #     print(f"Erro {e} no {os.path.basename(file_path)} em {int(t//60)}:{int(t%60)}")

    result = pd.concat(result,ignore_index=True)
    result.insert(0,"file",os.path.basename(file_path))

    return result

def RunDataProcessingFromParameterType1(file_path):
    start_timer = timeit.default_timer()
    print(f"Processando... {file_path}")

    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(file_path)

    # Importa do arquivo bruto
    model.df = pd.read_csv(model.raw_file)

    # Renomear colunas
    old_pattern = True if 'x_esquerda' in model.df.columns.tolist() else False
    # Ajusta os nomes das colunas para o padrão
    model.df = model.df.rename(columns={
        'id':model.id_column,
        'tipo':model.vehicle_type_column,
        'conf':model.conf_YOLO_column,
        'faixa':model.traffic_lane_column,
        'instante':model.instant_column,
    })
    
    model.df[model.frame_column] = model.df[model.frame_column].astype(int)
    model.df[model.id_column] = model.df[model.id_column].astype(int)
    model.df[model.vehicle_type_column] = model.df[model.vehicle_type_column].astype("category")
    model.df[model.conf_YOLO_column] = model.df[model.conf_YOLO_column].astype(float)
    model.df[model.traffic_lane_column] = model.df[model.traffic_lane_column].astype(int)
    model.df[model.instant_column] = model.df[model.instant_column].astype(float)

    if not old_pattern:
        model.df = model.df.rename(columns={
            'x1':model.p1_x_bb_column,
            'y1':model.p1_y_bb_column,
            'x2':model.p2_x_bb_column,
            'y2':model.p2_y_bb_column,
        })
    else:
        model.df = model.df.rename(columns={
            'x_esquerda':model.p1_x_bb_column,
            'y_superior':model.p1_y_bb_column,
            'x_largura':model.vehicle_length_column,
            'y_altura':model.vehicle_width_column,
        })

        model.df[model.p2_x_bb_column] = model.df[model.p1_x_bb_column] + model.df[model.vehicle_length_column]
        model.df[model.p2_y_bb_column] = model.df[model.p1_y_bb_column] + model.df[model.vehicle_width_column]
    
    # Ajuste do tipo de veículo
    for id in model.df[model.id_column].unique():
        model.df.loc[model.df[model.id_column]==id,model.vehicle_type_column] = model.FindVehicleType(id)

    # Converter variáveis de posição e distancia de pixels para metro
    model.df[model.p1_x_bb_column] = model.mpp*model.df[model.p1_x_bb_column]
    model.df[model.p1_y_bb_column] = model.mpp*model.df[model.p1_y_bb_column]
    model.df[model.p2_x_bb_column] = model.mpp*model.df[model.p2_x_bb_column]
    model.df[model.p2_y_bb_column] = model.mpp*model.df[model.p2_y_bb_column]
    
    # Cálculo das dimensões do veículo
    model.df[model.vehicle_length_column] = model.df[model.p2_x_bb_column]-model.df[model.p1_x_bb_column]
    model.df[model.vehicle_width_column] = model.df[model.p2_y_bb_column]-model.df[model.p1_y_bb_column]

    # Rotação horizontal se necessário
    if model.flip_h:
        # Coordenada horizontal à esquerda
        model.df[model.p1_x_bb_column] = model.video_width - model.df[model.p1_x_bb_column] - model.df[model.vehicle_length_column]
        # Recalculo do ponto à direita
        model.df[model.p2_x_bb_column] = model.df[model.p1_x_bb_column] + model.df[model.vehicle_length_column]
    
    if model.flip_v:
        # Coordenada horizontal à esquerda
        model.df[model.p1_y_bb_column] = model.video_heigth - model.df[model.p1_y_bb_column] - model.df[model.vehicle_width_column]
        # Recalculo do ponto à direita
        model.df[model.p2_y_bb_column] = model.df[model.p1_y_bb_column] + model.df[model.vehicle_width_column]
    
    # Cálculo do centroide x e y
    model.df[model.y_centroid_column] = (model.df[model.p1_y_bb_column] + model.df[model.vehicle_width_column]*0.5)
    model.df[model.x_centroid_column] = model.df[model.p1_x_bb_column] + model.df[model.vehicle_length_column]*0.5
    
    # Ajuste do número da faixa
    model.df = gpd.GeoDataFrame(model.df,
                                geometry=gpd.points_from_xy(model.df[model.x_centroid_column],model.df[model.y_centroid_column]),
                                crs="EPSG:31984")

    model.df = model.df.overlay(model.traffic_lane_polygon.rename(columns={"id":"tl_polygon"})[["tl_polygon","geometry"]],how='union')
    model.df[model.traffic_lane_column] = model.df["tl_polygon"]
    model.df = pd.DataFrame(model.df.drop(columns=["tl_polygon","geometry"]))

    # Cálculo de posições estrátégicas longitudinais o veículo
    # Fundo do veículo
    model.df[model.x_tail_column] = model.df[model.p1_x_bb_column]
    # Frente do veículo
    model.df[model.x_head_column] = model.df[model.p2_x_bb_column]

    # Arredonda o tempo
    model.df[model.instant_column] = model.df[model.instant_column].round(4)
    # Id geral, combinando id e tempo
    model.df[model.id_column] = model.df[model.id_column].astype(int)
    model.df[model.frame_column] = model.df[model.frame_column].astype(int)
    model.df[model.global_id_column] = model.df[model.id_column].astype(str) + '@' + model.df[model.frame_column].astype(str)

    # Remove valores com baixa incidência
    model.RemoveLowIncidence()
    # Calcula da velocidade e aceleração
    model.SpeedAndAccDetector()
    # Criar frames interpolados
    df_new = model.GhostFramesGenerator(model.df[model.id_column].unique(),step=1)
    model.df = pd.concat([model.df,df_new],ignore_index=True)
    model.df = model.df.sort_values(by=[model.frame_column,model.traffic_lane_column,model.x_centroid_column])

    model.df.to_csv(model.processed_file,index=False)
    print("Fim da execussão",model.processed_file)
    
    stop_timer = timeit.default_timer()
    count_timer = stop_timer - start_timer
    print(f"\tDuração: {int(count_timer//60)}min:{int(count_timer%60)}s")

if __name__=="__main__":
    
    root_json_path = "data/json"
    all_files = os.listdir(root_json_path)

    for file_name  in all_files:
        try:
            file_path = os.path.join(root_json_path,file_name)
            RunDataProcessingFromParameterType1(file_path)
        except Exception as e:
            print(e)
            

    # root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis"
    # output_folder = "data/hd4"
    # run = Main()
    # run.WorkflowPattern(
    #     root_path=root_path,
    #     output_folder=output_folder,
    #     prefix="Hd4_",
    #     func=RunHd4Analysis)
    
    # # Concatenar resumo
    # df = []
    # all_files = os.listdir(output_folder)
    # for f in all_files:
    #     df_ = pd.read_csv(os.path.join(output_folder,f))
    #     df.append(df_)
    # df = pd.concat(df,ignore_index=True)
    # df.to_excel("data/summary/hd4_23_04_25.xlsx",index=False)

    # output_folder = "data/hd1"
    # run = Main()
    # run.WorkflowPattern(
    #     root_path=root_path,
    #     output_folder=output_folder,
    #     prefix="Hd1_",
    #     func=RunHd1Analysis)
    
    # # Concatenar resumo
    # df = []
    # all_files = os.listdir(output_folder)
    # for f in all_files:
    #     df_ = pd.read_csv(os.path.join(output_folder,f))
    #     df.append(df_)
    # df = pd.concat(df,ignore_index=True)
    # df.to_excel("data/summary/hd1_23_04_25.xlsx",index=False)

    

    