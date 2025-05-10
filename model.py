from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import timeit
import json
import os

class  YoloMicroscopicDataProcessing:
    def __init__(self):
        # DEFAULT COLUMNS
        # Ids and time
        self.global_id_column = 'global_id'
        self.id_column = 'id'
        self.instant_column = 'instant'
        self.frame_column = 'frame'
        self.conf_YOLO_column = 'conf_YOLO'

        # Position in the plane
        self.y_centroid_column = 'y'
        self.x_centroid_column = 'x'
        self.x_head_column = 'head'
        self.x_tail_column = 'tail'
        self.p1_x_bb_column = 'p1xbb'
        self.p1_y_bb_column = 'p1ybb'
        self.p2_x_bb_column = 'p2xbb'
        self.p2_y_bb_column = 'p2ybb'

        # Moviment
        self.x_instant_speed_column = 'x_instant_speed'
        self.y_instant_speed_column = 'y_instant_speed'
        self.instant_speed_column = 'instant_speed'

        self.x_instant_acc_column = 'x_instant_acc'
        self.y_instant_acc_column = 'y_instant_acc'
        self.instant_acc_column = 'instant_acc'

        # Vehicle itself
        self.vehicle_type_column = 'vehicle_type'
        self.vehicle_length_column = 'vehicle_length'
        self.vehicle_width_column = 'vehicle_width'

        # Field conditions
        self.state_column = 'state'
        self.queue_position_column = 'queue_position'
        self.traffic_lane_column = 'traffic_lane'
        self.virtual_traffic_lane_column = 'virtual_traffic_lane'

        #-----------------------------------------------------------------------
        # GLOBAL PARAMETERS
        # Specific types of vehicles
        self.vehicle_type_list = ['Carro','Moto','Onibus','Van','Caminhao','Bicicleta']
        # Other classes
        self.other_type_list = ['Pessoa']
        # General categorys of vehicle
        self.vehicle_category_list = {
            'four_wheel':['Carro','Onibus','Van','Caminhao'],
            'two_wheel':['Moto','Bicicleta'],
            'passenger':['Carro','Van','Onibus'],
            'heavy':['Caminhao','Onibus'],
            'light':['Carro','Van'],
            'walk':['Pedestre']
        }
        # Parameter side offset vehicle dict
        self.side_offset_vehicle_dict = {
            'Moto':0,
            'Bicileta':0,
            'Carro':0,
            'Onibus':0,
            'Van':0,
            'Caminhao':0,
            'Pedestre':0
        }

        # Velocidade máxima absoleta para considerar que o veículo está parado em m/s
        self.global_stopped_speed_threshold = 1
        # FPS
        self.fps = 30
        # Flip
        self.flip_h = False
        self.flip_v = False

        #-----------------------------------------------------------------------

        # DATA PROCESSING
        # Input data format files
        self.read_format_list = {
            'csv':pd.read_csv,
            'xlsx':pd.read_excel
        }

        # Origin file path
        self.file_path = 'file.csv'
        # File format
        self.file_format = self.file_path.split('.')[-1]
        # Editable file
        self.df = ''

        #-----------------------------------------------------------------------

        # INPUT PARAMETERS
        self.mpp = 1
        self.video_width = 1920
        self.video_heigth = 1080
        self.motobox_start_section = 0
        self.motobox_end_section = 10000
        self.width_virtual_lane = 1
        self.virtual_lane_lim = []
        self.green_open_time = []
        self.image_reference = ""
        self.raw_file = ""
        self.processed_file = ""
        self.parameter_file = ""
        self.motorcycle_waiting_area_polygon = pd.DataFrame(columns=["id","coords"])
        self.traffic_lane_polygon = pd.DataFrame(columns=["id","coords"])

        #-----------------------------------------------------------------------

        # OTHER DEFAULT LISTS
        # Lista de cores para as classes
        self.colors_type_list = {
            'Carro':'red',
            'Moto':'blue',
            'Onibus':'green',
            'Van':'brown',
            'Bicicleta':'yellow',
            'Caminhao':'purple'
        }

    def ExportParameter(self,file_path:str,logs=True):
        '''
        Exporta os parâmetros do video como um arquivo csv para ser lido futuramente

        Salvar antes de transformar os parâmetros de pixel em metros e após
        flipar o vídeo horizontalmente, pois isso altera alguns parâmetros

        A função não retorna valor
        '''

        d = {
                'mpp':[self.mpp],
                'video_width':[self.video_width],
                'video_heigth':[self.video_heigth],
                'motobox_start_section':[self.motobox_start_section],
                'motobox_end_section':[self.motobox_end_section],
                'width_virtual_lane':[self.width_virtual_lane],
                'virtual_lane_lim':[self.virtual_lane_lim],
                'green_open_time':[self.green_open_time],
                'image_reference':[self.image_reference]
            }
        config_parameter = pd.DataFrame.from_dict(d)
        config_parameter.to_csv(file_path,index=False)

        if logs:
            print(f'Parâmetro exportado de: {file_path}')

    def ParameterConverter(self):
        '''
        Essa função converte os inputs atribuidos de pixels para metros.
        Utilizar após atribuir manualmente os inputs (primeiro tratamento) ou
        após importar os inputs com a função "ImportParameter"

        No primeiro caso, onde o arquivo de parâmetros ainda será salvo, deve-se
        fazer isso antes de executar essa função, afim de que os dados sejam
        salvos em pixels (possibilitando a validação simples) e não em metros

        Exige que o parâmetros mpp esteja atualizado

        Em ambos os casos, os inputs ainda devem estar em formato de pixels

        Converte para metro os parâmetos:
        self.video_width
        self.video_heigth
        self.motobox_start_section
        self.motobox_end_section
        self.virtual_lane_lim

        A função não retorna valor
        '''

        self.video_width = self.video_width*self.mpp
        self.video_heigth = self.video_heigth*self.mpp
        self.motobox_start_section = self.motobox_start_section*self.mpp
        self.motobox_end_section = self.motobox_end_section*self.mpp
        self.virtual_lane_lim = [[[i[0]*self.mpp,i[-1]*self.mpp] for i in j] for j in self.virtual_lane_lim]

    def ImportParameter(self,file_path:str,logs=False):
        '''
        Importa os parâmetros do video como um arquivo csv e realiza as conversoes necessárias

        A função não retorna valor
        '''

        config_parameter = pd.read_csv(file_path)

        # Parâmetros lidos diretamente
        self.mpp = config_parameter['mpp'].tolist()[0]
        self.video_width = config_parameter['video_width'].tolist()[0]
        self.video_heigth = config_parameter['video_heigth'].tolist()[0]
        self.motobox_start_section = config_parameter['motobox_start_section'].tolist()[0]
        self.motobox_end_section = config_parameter['motobox_end_section'].tolist()[0]
        self.width_virtual_lane = config_parameter['width_virtual_lane'].tolist()[0]
        self.image_reference = config_parameter['image_reference'].tolist()[0]

        # Parâmetros salvos como listas em python
        self.virtual_lane_lim = eval(config_parameter['virtual_lane_lim'].tolist()[0])
        self.green_open_time = eval(config_parameter['green_open_time'].tolist()[0])

        if logs:
            print(f'Parâmetro importado de: {file_path}')
        
        # Conver os parâmetros
        self.ParameterConverter()

    def MotorcycleWaitingAreaCoordsFromSEH0(
        self,
        start_lim:float,
        end_lim:float,
        height_lim:float,
        ):

        """
        Organiza o retângulo do motobox com base nos limites de
        início e fim do motobox e considerando a extensão inteira 
        da altura da imagem
        Retorna as coordenadas do polígono
        """

        mwa_polygon_coords = [
                [start_lim,0],
                [start_lim,height_lim],
                [end_lim,height_lim],
                [end_lim, 0],
                [start_lim,0],
            ]
        
        return mwa_polygon_coords
    
    def TrafficLaneCoordsFromLimits(
        self,
        upper_lim_coords:list,
        lower_lim_coords:list
        ):
        """
        Converte o formato de limites das faixas em um formato
        compatível com a criação de um polígono
        """

        tlc_polygon_coords = upper_lim_coords+lower_lim_coords[::-1]
        
        return tlc_polygon_coords
    
    def SavePolygonRegionImg(self,flip_h_img):
        """
        Salva uma imagem com os poligonos existentes
        """
        # Estrutura o nome para salvar o arquivo
        ext_file = os.path.splitext(self.image_reference)[-1]
        savefig_file = self.image_reference.replace(ext_file,f"_region{ext_file}")

        # Lê a imagem de fundo e invere horizontalmente se for conveniente
        img = plt.imread(self.image_reference)
        if flip_h_img:
            img = np.flip(img,axis=1)

        # Cria o ax
        fig, ax = plt.subplots()
        # Remove os eixos
        ax.axis('off')
        # Coloca a imagem de fundo
        ax.imshow(img)
        # Toma as coordenadas do motobox e transforma para o tamanho da imagem
        x_mwa,ymwa = shapely.affinity.scale(
            self.motorcycle_waiting_area_polygon,
            xfact=1/self.mpp,
            yfact=1/self.mpp,
            origin=(0,0,0)
            ).exterior.xy
        # Cria a região do motobox
        ax.fill_between(x_mwa,ymwa,alpha=0.5)
        # Define por linhas
        ax.plot(x_mwa,ymwa)
        
        # Salva a imagem sem bordas
        plt.savefig(savefig_file,bbox_inches='tight',pad_inches=0)
        # Limpa as configurações e plots feitos
        plt.clf()

        return True
    
    def ScalePxToMeterPolygon(self,geom):
        """
        Escala a geometia de pixels para metros
        """
        geom = shapely.affinity.scale(geom,xfact=self.mpp,yfact=self.mpp,origin=(0,0))

        return geom

    def CreateJSON(self,file_path):
        """
        Export o arquivo ".json" padronizado
        Retorna True se todas as operações forem realizadas
        """

        cfg = {
            "raw_file": f"data/raw/{os.path.basename(self.raw_file)}",
            "processed_file": f"data/processed/{os.path.basename(self.processed_file)}",
            "parameter_file": f"data/parameter/{os.path.basename(self.parameter_file)}",
            "mpp":self.mpp,
            "video_width": int(self.video_width/self.mpp),
            "video_heigth": int(self.video_heigth/self.mpp),
            "motobox_start_section":int(self.motobox_start_section/self.mpp),
            "motobox_end_section":int(self.motobox_end_section/self.mpp),
            "motorcycle_waiting_area_polygon":self.motorcycle_waiting_area_polygon[["id","coords"]].to_dict(),
            "width_virtual_lane":self.width_virtual_lane,
            "virtual_lane_lim":[[[int(i[0]/self.mpp),int(i[-1]/self.mpp)] for i in j] for j in self.virtual_lane_lim],
            "traffic_lane_polygon":self.traffic_lane_polygon[["id","coords"]].to_dict(),
            "green_open_time":self.green_open_time,
            "image_reference": f"data/image/{os.path.basename(str(self.image_reference))}",  # Alguns dados foram colocados como só o número do frame
            "flip_h":self.flip_h,
            "flip_v":self.flip_v,
        }

        # Salvar arquivo        
        with open(file_path,'w',encoding="utf-8",errors="ignore") as f:  
            json.dump(cfg,f,indent=4)
    
    def PostProcessing1(self):
        """
        Pós processamento aplicálvel a depender do uso
        Esta remove observações (frames) de bicicletas e pedestres
        E ids (trajetória completa) se não tiver faixa associada em todos os frames
        Não retorna valor, altera diretamente o self.df
        """
        # Remove observações de bicicletas e pedestres
        self.df = self.df[-self.df[self.vehicle_type_column].isin(["Bicicleta","Pedestre"])]
        
        # Remove veículos (ids) que só apresentam np.nan em self.traffic_lane_column
        # df_traffic_lane_agg = self.df.groupby(self.id_column).agg({self.traffic_lane_column:lambda values:values.isna().all()}).reset_index(drop=False)
        # df_traffic_lane_agg = df_traffic_lane_agg[-df_traffic_lane_agg[self.traffic_lane_column]]
        # self.df = self.df[self.df[self.id_column].isin(df_traffic_lane_agg[self.id_column].tolist())]

        self.df = self.df.dropna(subset=self.traffic_lane_column)

        self.df = self.df.reset_index(drop=True)

    def ImportFromJSON(self,file_path,post_processing=None):
        """
        Puxa os dados e metadados de um arquivo .json padronizado
        Retorna True se todas as operações forem realizadas
        """
        # Lê o json
        with open(file_path,encoding="utf-8",errors="ignore") as f:
            cfg = json.load(f)

        # Verifica se o json possui as colunas mínimas
        req_keys = [
            'processed_file',
            'mpp',
            'video_width',
            'video_heigth',
            'motorcycle_waiting_area_polygon',
            'width_virtual_lane',
            'traffic_lane_polygon',
            'green_open_time',
            'image_reference'
        ]
        for col in req_keys:
            if col not in cfg.keys():
                raise ValueError(f"O parâmetro {col} não consta no arquivo '.json'.")
        
        # Compatibilização de padrões
        #----------------------------------------------------------------------------------
        # Verifica se o campo "motorcycle_waiting_area_polygon", se for vazio,
        # constrói o polígono por meio de uma informação obsoleta
        # "motobox_start_section" e "motobox_end_section"
        if cfg["motorcycle_waiting_area_polygon"]=={"id": {},"coords": {}}:
            # Coordenadas organizadas com base nos limites do motobox
            mwa_polygon_coords = self.MotorcycleWaitingAreaCoordsFromSEH0(
                cfg["motobox_start_section"],
                cfg["motobox_end_section"],
                cfg["video_heigth"])
            
            # Atualiza dicionário como um outro dicionário
            cfg["motorcycle_waiting_area_polygon"] = dict(zip(["id","coords"],[["1"],[mwa_polygon_coords]]))
        
        # Verifica se o campo "traffic_lane_polygon", se for vazio
        # constrói o polígono por meio de uma informação obsoleta
        # "virtual_lane_lim"
        if cfg["traffic_lane_polygon"]=={"id": {},"coords": {}}:
            traffic_lane_polygon_coords = [
                self.TrafficLaneCoordsFromLimits(i,j)
                for i,j in zip(cfg["virtual_lane_lim"][:-1],cfg["virtual_lane_lim"][1:])]
            num_traffic_lanes = len(traffic_lane_polygon_coords)
            id_traffic_lane = [str(i) for i in  range(1,num_traffic_lanes+1)]
            # Atualiza dicionário
            cfg["traffic_lane_polygon"] = dict(zip(["id","coords"],[id_traffic_lane,traffic_lane_polygon_coords]))
        
        # Fim das compatibilizações e ajustes
        #----------------------------------------------------------------------------------

        # Atribui o fator de escala
        self.mpp = cfg["mpp"]
        # Atribui o motobox como um geodataframe
        self.motorcycle_waiting_area_polygon = gpd.GeoDataFrame(
            cfg["motorcycle_waiting_area_polygon"],
            geometry=[shapely.Polygon(i) for i in cfg["motorcycle_waiting_area_polygon"]["coords"]],
            crs="EPSG:31984")
        self.motorcycle_waiting_area_polygon["geometry"] = self.motorcycle_waiting_area_polygon["geometry"].apply(self.ScalePxToMeterPolygon)
        
        # Atribui as regiões das faixas de tráfego
        self.traffic_lane_polygon = gpd.GeoDataFrame(
            cfg["traffic_lane_polygon"],
            geometry=[shapely.Polygon(i) for i in cfg["traffic_lane_polygon"]["coords"]],
            crs="EPSG:31984")
        self.traffic_lane_polygon["geometry"] = self.traffic_lane_polygon["geometry"].apply(self.ScalePxToMeterPolygon)

        # Atribuir dimensões do vídeo
        self.video_heigth = cfg["video_heigth"]*self.mpp
        self.video_width = cfg["video_width"]*self.mpp
        
        # Características do motobox (obsoleto)
        self.motobox_start_section = cfg["motobox_start_section"]*self.mpp
        self.motobox_end_section = cfg["motobox_end_section"]*self.mpp

        # Limites das faixas (obsoleto)
        self.virtual_lane_lim = cfg['virtual_lane_lim']
        self.virtual_lane_lim = [[[i[0]*self.mpp,i[-1]*self.mpp] for i in j] for j in self.virtual_lane_lim]
        
        # Largura teórica do corredor virtual
        self.width_virtual_lane = cfg["width_virtual_lane"]

        # Instantes em que o verde abre
        self.green_open_time = cfg['green_open_time']

        # Nomes dos arquivos
        self.raw_file = cfg["raw_file"]
        self.processed_file = cfg["processed_file"]
        self.parameter_file = cfg["parameter_file"]
        self.image_reference = cfg["image_reference"]

        # Inversões já aplicadas aos dados
        self.flip_h = cfg["flip_h"]
        self.flip_v = cfg["flip_v"]

        # Tenta atribuir o arquivo
        try:
            self.df = pd.read_csv(self.processed_file)
            self.df[self.id_column] = self.df[self.id_column].astype(int)
            self.df[self.frame_column] = self.df[self.frame_column].astype(int)
            # self.df[self.traffic_lane_column] = self.df[self.traffic_lane_column].astype(int)

            if post_processing!=None:
                post_processing()
        except:
            self.df = pd.DataFrame()
            print("Trajetórias ainda não processadas!")
        finally:
            # Atualiza o json
            with open(file_path,'w',encoding="utf-8",errors="ignore") as f:  
                json.dump(cfg,f,indent=4)

    def RemoveLowIncidence(self,threshold:int=20):
        '''
        Remove os ids com baixa incidência, abaixo do limite definido
        Não retorna valor
        '''
        df_incidence = self.df.groupby(self.id_column).count().sort_values(self.frame_column)[self.frame_column]
        mask = self.df[self.id_column].isin(df_incidence[df_incidence>=threshold].index.tolist())
        self.df = self.df[mask]

    def GhostFramesGenerator(self,id_vehicle_list:list,range_frame:tuple=None,step=1,max_abs_generator=300):
        '''
        Cria "Ghost Frames" para os dados numéricos nas colunas especificadas
        Em frames identificados como ausentes
        Interpolação linear dos parâmetros
        Considera-se somente os vizinhos imediatamente mais próximos
        '''

        # Filtro de dados de id e frame (se especificado)
        if range_frame==None:
            df_vehicles = self.df[self.df[self.id_column].isin(id_vehicle_list)]
        else:
            df_vehicles = self.df[(self.df[self.id_column].isin(id_vehicle_list)) & (self.df[self.frame_column].between(range_frame[0],range_frame[1]))]

        df_ghost_frames = pd.DataFrame(columns=df_vehicles.columns)

        for id_vehicle in df_vehicles[self.id_column].unique():
            # print(id_vehicle,'Inicio')
            df_analysis = df_vehicles[df_vehicles[self.id_column]==id_vehicle]
            # O gradiente esperado de um trecho contínuo é de 1
            gradient_frames = np.gradient(df_analysis[self.frame_column])

            # Se o único valor de gradiente for igual o step, implica que é contínuo
            if (np.unique(gradient_frames)==np.array([step])).all():
                # Não realiza-se alterações
                # print(id_vehicle,'Fim')
                pass

            # Se não, há frames a serem gerados, calcula-se o necessário
            else:
                # Colunas numéricas a serem ajustadas
                exception_list = [
                    self.frame_column,
                    self.id_column,
                    self.conf_YOLO_column,
                ]
                generator_columns = [i for i in df_analysis.dtypes[df_analysis.dtypes.isin([np.dtype(k) for k in ['int64','float64','int32','float32']])].index if i not in exception_list]

                # Frames necessários
                max_frame = df_analysis[self.frame_column].max()
                min_frame = df_analysis[self.frame_column].min()
                required_frames = [j for j in range(min_frame,max_frame+1,step) if j not in df_analysis[self.frame_column].tolist()]

                # Se a quantidade de frames a ser gerada for maior que
                # "max_relative_generator", não será computado devido ao esforço
                # grande e provável grande lacuna existente
                # print(len(required_frames),len(required_frames)/(max_frame-min_frame))

                if len(required_frames)>max_abs_generator:
                    # print(id_vehicle,'Fim (não processado)')
                    pass
                else:
                    # Dataframe de ghost frames
                    df_gf = pd.DataFrame(columns=df_analysis.columns)
                    df_gf[self.frame_column] = required_frames
                    df_gf[self.vehicle_type_column] = [df_analysis[self.vehicle_type_column].mode().iloc[0]]*len(required_frames)
                    df_gf[self.id_column] = [id_vehicle]*len(required_frames)
                    df_gf[self.conf_YOLO_column] = [0]*len(required_frames)

                    for col in generator_columns:
                        # Cria a função da variável "col" em função dos frames originais
                        df_ = df_analysis[[self.frame_column,col]].sort_values(self.frame_column)

                        # Filtra somente os frames próximo aos gaps para tornar a função mais rápida
                        required_OG_frames = []
                        for i in required_frames:
                            inf = df_analysis[df_analysis[self.frame_column]<i][self.frame_column].max()
                            sup = df_analysis[df_analysis[self.frame_column]>i][self.frame_column].min()

                            if not inf in required_OG_frames:
                                required_OG_frames.append(inf)
                            if not sup in required_OG_frames:
                                required_OG_frames.append(sup)

                        df_ = df_[df_[self.frame_column].isin(required_OG_frames)].values.tolist()

                        # Cria a função
                        ghost_frame_func = PolygonalToFunction(df_)

                        # Gera os ghost frames
                        df_gf[col] = df_gf[self.frame_column].apply(ghost_frame_func)

                        # Caso a coluna seja "traffic_lane", arredonda-se o resultado
                        if col==self.traffic_lane_column:
                            df_gf[col] = df_gf[col].round(0).astype(int)
                        if col==self.instant_column:
                            df_gf[col] = df_gf[col].round(2)

                    # Ajuste da coluna id_global
                    df_gf[self.global_id_column] = df_gf[self.id_column].astype(str)+'@'+df_gf[self.instant_column].round(2).astype(str)
                    # Concatena os frames gerados com os originais
                    df_ghost_frames = pd.concat([df_ghost_frames,df_gf],ignore_index=True)
                    # print(id_vehicle,'Fim')

        # Retorna o dataframe do id, com a geração de frames ou não a depender da necessidade
        return df_ghost_frames

    def SideVehicle(
            self,
            id_vehicle:int,
            frame:int,
            overlap_lon:float=0,
            overlap_lat:float=0,
            side:str='both',
            ignore_vehicle_types_list:list=None,
            max_lat_dist:float=None):
        '''
        O sentido adotado para descrever esquerda e direita refere-se ao sentido do
        vídeo gradado com o tráfego deslocando-se de oeste para leste

        O parâmetro overlap reduzr o tamanho do boundbox real do próprio veículo se for positivo
        Se for negativo, aumenta o bouding box do veículo cujo o id foi passado
        Funciona para considerar sobreposições exitentes lateralmente "overlarp"

        Retorna um dataframe com os veículos na lateral com as colunas
        "lateral_distance_between_vehicles" e "side"
        '''
        if ignore_vehicle_types_list==None:
            ignore_vehicle_types_list = []

        # Ajuste do instante do tempo
         # = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]

        motorcycle = self.df[(self.df[self.frame_column]==frame) & (self.df[self.id_column]==id_vehicle)]
        vehicles = self.df[self.df[self.frame_column]==frame]

        # Limites Laterais
        side_lim_1 = motorcycle[self.p1_y_bb_column].iloc[0]+overlap_lat
        side_lim_2 = motorcycle[self.p2_y_bb_column].iloc[0]-overlap_lat
        # Limites Longitudinais
        left_lim = motorcycle[self.p1_x_bb_column].iloc[0]+overlap_lon
        right_lim = motorcycle[self.p2_x_bb_column].iloc[0]-overlap_lon

        if side in ['right','left','both']:
            pass
        else:
            raise ValueError(f'O parâmetro side = "{side}" é inválido.')

        # Filtros
        if (side=='left') or (side=='both'):
            df_analysis_side_1 = vehicles[(vehicles[self.p2_y_bb_column]<=side_lim_1) & (vehicles[self.p2_x_bb_column]>=left_lim) & (vehicles[self.p1_x_bb_column]<=right_lim)]
            df_analysis_side_1['lateral_distance_between_vehicles'] = side_lim_1 - df_analysis_side_1[self.p2_y_bb_column] - overlap_lat
            df_analysis_side_1['side'] = ['left']*len(df_analysis_side_1)
        if (side=='right') or (side=='both'):
            df_analysis_side_2 = vehicles[(vehicles[self.p1_y_bb_column]>=side_lim_2) & (vehicles[self.p2_x_bb_column]>=left_lim) & (vehicles[self.p1_x_bb_column]<=right_lim)]
            df_analysis_side_2['lateral_distance_between_vehicles'] = df_analysis_side_2[self.p1_y_bb_column] - side_lim_2 - overlap_lat
            df_analysis_side_2['side'] = ['right']*len(df_analysis_side_2)
        if side=='both':
            df_analysis = pd.concat([df_analysis_side_1,df_analysis_side_2],ignore_index=True)
        else:
            if side=='left':
                df_analysis = df_analysis_side_1
            if side=='right':
                df_analysis = df_analysis_side_2
            else:
                raise ValueError('Erro!')

        # Filtra os tipos de veículos, removendo os sem interesse
        df_analysis = df_analysis[-df_analysis[self.vehicle_type_column].isin(ignore_vehicle_types_list)]
        # Filtra veículos muito distantes lateralmente
        if max_lat_dist!=None:
            df_analysis = df_analysis[df_analysis["lateral_distance_between_vehicles"]<=max_lat_dist]

        return df_analysis

    def SpeedAndAccDetector(self):
        '''
        Estima a velocidade para cada veículo a cada instante de tempo utilizando
        o "np.gradient" do módulo numpy

        A função não retorna valor
        '''
        df_speed_acc = pd.DataFrame()

        id_vehicle_list = self.df[self.id_column].unique().tolist()
        for id_vehicle in id_vehicle_list:
            # Dados do veiculo
            df_analysed = self.df[self.df[self.id_column]==id_vehicle].sort_values(self.instant_column)

            # Posição e tempo organizados temporalmente
            x = df_analysed[self.x_centroid_column]
            y = df_analysed[self.y_centroid_column]
            t = df_analysed[self.instant_column]

            # Calculo da velocidade
            x_instant_speed = np.gradient(x,t,edge_order=2)
            y_instant_speed = np.gradient(y,t,edge_order=2)

            # if smooth_speed:
            #     x_instant_speed = savgol_filter(x_instant_speed,5,1)
            #     y_instant_speed = savgol_filter(y_instant_speed,5,1)

            # Calculo da aceleração
            x_instant_acc = np.gradient(x_instant_speed,t,edge_order=2)
            y_instant_acc = np.gradient(y_instant_speed,t,edge_order=2)

            # Associação com o veículo
            df_analysed[self.x_instant_speed_column] = x_instant_speed
            df_analysed[self.y_instant_speed_column] = y_instant_speed
            df_analysed[self.x_instant_acc_column] = x_instant_acc
            df_analysed[self.y_instant_acc_column] = y_instant_acc

            # Cálculo do módulo
            df_analysed[self.instant_speed_column] = ((df_analysed[self.x_instant_speed_column]**2) + (df_analysed[self.y_instant_speed_column]**2))**0.5
            df_analysed[self.instant_acc_column] = ((df_analysed[self.x_instant_acc_column]**2) + (df_analysed[self.y_instant_acc_column]**2))**0.5

            df_speed_acc = pd.concat([df_speed_acc,df_analysed[[
                self.global_id_column,
                self.x_instant_speed_column,self.y_instant_speed_column,self.instant_speed_column,
                self.x_instant_acc_column,self.y_instant_acc_column,self.instant_acc_column
                ]]],ignore_index=True)

        self.df = self.df.merge(df_speed_acc,on=self.global_id_column,how='left')

    def SmoothingGaussianFilter(self,step=3,sigma=3):
        '''
        Suaviza as variáveis de posição, velocidade e aceleleração com base em
        procedimendo especificicado dentro da função
        Descrição:
        * Redução para a frequência especificada pelo step, no caso Freq = FreqAtual/step
        * Suavização da posição dos pontos limites do veículo pelo método gausiano unidimensional com o sigma especificado
        * Recálculo das coordenadas de centróide, altura e largura
        * Recálculo da velocidade pelo np.gradient baseado no centroide
        * Suavização da velocidade pelo método gausiano unidimensional com o sigma especificado
        * Suavização da aceleração pelo  pelo np.gradient baseado no centroide
        * Suavização da aceleração pelo método gausiano unidimensional com o sigma especificado
        '''
        df_smooth = pd.DataFrame()
        frame_range_list = list(range(min(self.df[self.frame_column]),max(self.df[self.frame_column])+1,step))
        cols_keep = self.df.columns

        id_vehicle_list = self.df[self.id_column].unique().tolist()
        for id_vehicle in id_vehicle_list:
            # Dados do veiculo
            df_analysed = self.df[self.df[self.id_column]==id_vehicle].sort_values(self.instant_column)
            # Filtro para garantir que somente os frames certos sejam considerados
            df_analysed = df_analysed[df_analysed[self.frame_column].isin(frame_range_list)]

            # Instante de tempo
            t = df_analysed[self.instant_column]

            # Suavização do Ponto 1
            x1 = gaussian_filter1d(df_analysed[self.p1_x_bb_column],sigma=sigma)
            y1 = gaussian_filter1d(df_analysed[self.p1_y_bb_column],sigma=sigma)
            # Suavização do Ponto 2
            x2 = gaussian_filter1d(df_analysed[self.p2_x_bb_column],sigma=sigma)
            y2 = gaussian_filter1d(df_analysed[self.p2_y_bb_column],sigma=sigma)

            # Recálcula de valores de posição
            # Centroide
            xc = (x1+x2)*0.5
            yc = (y1+y2)*0.5
            # Comprimento
            length = abs(x1-x2)
            # Largura
            width = abs(y1-y2)
            # Head
            head = xc + (length/2)
            # Tail
            tail = xc - (length/2)

            # Recalculo da velocidade
            x_instant_speed = np.gradient(xc,t,edge_order=2)
            y_instant_speed = np.gradient(yc,t,edge_order=2)
            # Suavização da velocidade
            x_instant_speed = gaussian_filter1d(x_instant_speed,sigma=sigma)
            y_instant_speed = gaussian_filter1d(y_instant_speed,sigma=sigma)
            # Recalculo da aceleração
            x_instant_acc = np.gradient(x_instant_speed,t,edge_order=2)
            y_instant_acc = np.gradient(y_instant_speed,t,edge_order=2)
            # Suavização da aceleração
            x_instant_acc = gaussian_filter1d(x_instant_acc,sigma=sigma)
            y_instant_acc = gaussian_filter1d(y_instant_acc,sigma=sigma)

            # Associação com o veículo das variáveis atualizadas
            # Posição
            df_analysed[self.y_centroid_column] = yc
            df_analysed[self.x_centroid_column] = xc
            df_analysed[self.x_head_column] = head
            df_analysed[self.x_tail_column] = tail
            df_analysed[self.p1_x_bb_column] = x1
            df_analysed[self.p1_y_bb_column] = y1
            df_analysed[self.p2_x_bb_column] = x2
            df_analysed[self.p2_y_bb_column] = y2
            df_analysed[self.vehicle_length_column] = length
            df_analysed[self.vehicle_width_column] = width
            # Velocidade
            df_analysed[self.x_instant_speed_column] = x_instant_speed
            df_analysed[self.y_instant_speed_column] = y_instant_speed
            df_analysed[self.instant_speed_column] = ((df_analysed[self.x_instant_speed_column]**2) + (df_analysed[self.y_instant_speed_column]**2))**0.5
            # Aceleração
            df_analysed[self.x_instant_acc_column] = x_instant_acc
            df_analysed[self.y_instant_acc_column] = y_instant_acc
            df_analysed[self.instant_acc_column] = ((df_analysed[self.x_instant_acc_column]**2) + (df_analysed[self.y_instant_acc_column]**2))**0.5

            # Mesclar coms os outros id
            df_smooth = pd.concat([df_smooth,df_analysed[cols_keep]],ignore_index=True)

        df_smooth = df_smooth.sort_values([self.frame_column,self.traffic_lane_column,self.x_centroid_column])
        df_smooth = df_smooth.reset_index(drop=True)

        return df_smooth

    def SmoothingSavGolFilter(self,window_length,polyorder):
        '''
        Suaviza as variáveis de posição, velocidade e aceleleração com base em
        procedimendo especificicado dentro da função
        Descrição:
        * Redução para a frequência especificada pelo step, no caso Freq = FreqAtual/step
        * Suavização da posição dos pontos limites do veículo pelo método gausiano unidimensional com o sigma especificado
        * Recálculo das coordenadas de centróide, altura e largura
        * Recálculo da velocidade pelo np.gradient baseado no centroide
        * Suavização da velocidade pelo método gausiano unidimensional com o sigma especificado
        * Suavização da aceleração pelo  pelo np.gradient baseado no centroide
        * Suavização da aceleração pelo método gausiano unidimensional com o sigma especificado
        '''
        df_smooth = pd.DataFrame()
        frame_range_list = list(range(min(self.df[self.frame_column]),max(self.df[self.frame_column])+1,1))
        cols_keep = self.df.columns

        id_vehicle_list = self.df[self.id_column].unique().tolist()
        for id_vehicle in id_vehicle_list:
            # Dados do veiculo
            df_analysed = self.df[self.df[self.id_column]==id_vehicle].sort_values(self.instant_column)
            # Filtro para garantir que somente os frames certos sejam considerados
            df_analysed = df_analysed[df_analysed[self.frame_column].isin(frame_range_list)]

            # Instante de tempo
            t = df_analysed[self.instant_column]

            # Suavização do Ponto 1
            x1 = savgol_filter(df_analysed[self.p1_x_bb_column],window_length=window_length,polyorder=polyorder)
            y1 = savgol_filter(df_analysed[self.p1_y_bb_column],window_length=window_length,polyorder=polyorder)
            # Suavização do Ponto 2
            x2 = savgol_filter(df_analysed[self.p2_x_bb_column],window_length=window_length,polyorder=polyorder)
            y2 = savgol_filter(df_analysed[self.p2_y_bb_column],window_length=window_length,polyorder=polyorder)

            # Recálcula de valores de posição
            # Centroide
            xc = (x1+x2)*0.5
            yc = (y1+y2)*0.5
            # Comprimento
            length = abs(x1-x2)
            # Largura
            width = abs(y1-y2)
            # Head
            head = xc + (length/2)
            # Tail
            tail = xc - (length/2)

            # Recalculo da velocidade
            x_instant_speed = np.gradient(xc,t,edge_order=2)
            y_instant_speed = np.gradient(yc,t,edge_order=2)
            # Suavização da velocidade
            x_instant_speed = savgol_filter(x_instant_speed,window_length=window_length,polyorder=polyorder)
            y_instant_speed = savgol_filter(y_instant_speed,window_length=window_length,polyorder=polyorder)
            # Recalculo da aceleração
            x_instant_acc = np.gradient(x_instant_speed,t,edge_order=2)
            y_instant_acc = np.gradient(y_instant_speed,t,edge_order=2)
            # Suavização da aceleração
            x_instant_acc = savgol_filter(x_instant_acc,window_length=window_length,polyorder=polyorder)
            y_instant_acc = savgol_filter(y_instant_acc,window_length=window_length,polyorder=polyorder)

            # Associação com o veículo das variáveis atualizadas
            # Posição
            df_analysed[self.y_centroid_column] = yc
            df_analysed[self.x_centroid_column] = xc
            df_analysed[self.x_head_column] = head
            df_analysed[self.x_tail_column] = tail
            df_analysed[self.p1_x_bb_column] = x1
            df_analysed[self.p1_y_bb_column] = y1
            df_analysed[self.p2_x_bb_column] = x2
            df_analysed[self.p2_y_bb_column] = y2
            df_analysed[self.vehicle_length_column] = length
            df_analysed[self.vehicle_width_column] = width
            # Velocidade
            df_analysed[self.x_instant_speed_column] = x_instant_speed
            df_analysed[self.y_instant_speed_column] = y_instant_speed
            df_analysed[self.instant_speed_column] = ((df_analysed[self.x_instant_speed_column]**2) + (df_analysed[self.y_instant_speed_column]**2))**0.5
            # Aceleração
            df_analysed[self.x_instant_acc_column] = x_instant_acc
            df_analysed[self.y_instant_acc_column] = y_instant_acc
            df_analysed[self.instant_acc_column] = ((df_analysed[self.x_instant_acc_column]**2) + (df_analysed[self.y_instant_acc_column]**2))**0.5

            # Mesclar coms os outros id
            df_smooth = pd.concat([df_smooth,df_analysed[cols_keep]],ignore_index=True)

        df_smooth = df_smooth.sort_values([self.frame_column,self.traffic_lane_column,self.x_centroid_column])
        df_smooth = df_smooth.reset_index(drop=True)

        return df_smooth

    def QueueDetector(
        self,
        frame:int,
        ignore_vehicle_types_list_order:list=None,
        ignore_vehicle_types_list_distance:list=None,
        distance_between_vehicles_lim:float=7,
        ignore_ext_interference=True,
        ):
        '''
        Retorna os veículos em fila por faixa
        Se não tiver nenhum veículo, retona None
        '''

        if ignore_vehicle_types_list_order==None:
            ignore_vehicle_types_list_order = self.vehicle_category_list["two_wheel"] + self.vehicle_category_list["walk"]
        if ignore_vehicle_types_list_distance==None:
            ignore_vehicle_types_list_distance = self.vehicle_category_list["two_wheel"] + self.vehicle_category_list["walk"]
        
        # Dados válidos
        df_analysed = self.df[self.df[self.frame_column]==frame]
        df_analysed = df_analysed[-df_analysed[self.vehicle_type_column].isin(ignore_vehicle_types_list_order)]
        df_analysed = df_analysed.sort_values(by=self.x_head_column,ascending=False)

        if ignore_ext_interference:
            df_analysed = df_analysed.dropna(subset=self.traffic_lane_column)
        # Se não tiver nenhum veículo, retona o dataframe vazio
        if df_analysed.empty:
            return df_analysed

        # Lista de filas
        df_queue = pd.DataFrame()

        while not df_analysed.empty:
            # Dados do primeiro veículo
            id_vehicle = df_analysed.iloc[0][self.id_column]
            vehicle = df_analysed[df_analysed[self.id_column]==id_vehicle]
            first_traffic_lane_vehicle = df_analysed.iloc[0][self.traffic_lane_column]
            queue_position = 1
            vehicle[self.queue_position_column] = [queue_position]
            vehicle['first_traffic_lane_vehicle'] = [first_traffic_lane_vehicle]
            vehicle['distance_between_vehicles'] = [0]

            # Em situações dinâmicas, ou quando veículos estão muito distantes
            # sem motivo (motos entre veículos por exemplo), não considera-se uma
            # fila efetivamente, portanto, uma uma mesma faixa podem haver "filas"
            # 1 para o caso de ser a fila mais a frente e 0 para as demais
            if len(df_queue)>0:
                if first_traffic_lane_vehicle in df_queue[self.traffic_lane_column].unique().tolist():
                    first_group = 0
                else:
                    first_group = 1
            else:
                first_group = 1
            vehicle['first_group'] = [first_group]

            # União dos dados
            df_queue = pd.concat([df_queue,vehicle],ignore_index=True)

            # Veículo de trás mais próximo na projeção
            behind_vehicle = self.FirstVehicleBehind(
                id_vehicle=id_vehicle,
                frame=frame,
                ignore_vehicle_types_list=ignore_vehicle_types_list_distance)

            while not behind_vehicle.empty:
                # São considerados em fila, se os veículos não estiverem muito distantes entre si
                if behind_vehicle.iloc[0]['distance_between_vehicles']<=distance_between_vehicles_lim:
                    # Dados dos veiculos atrás
                    id_vehicle = behind_vehicle.iloc[0][self.id_column]
                    vehicle = behind_vehicle

                    # Se for moto, conta para a "união da fila", mas não como uma posição efetivamente
                    if vehicle.iloc[0][self.vehicle_type_column] in self.vehicle_category_list['four_wheel']:
                        queue_position = queue_position + 1
                        vehicle[self.queue_position_column] = [queue_position]
                    else:
                        vehicle[self.queue_position_column] = [-1]
                    vehicle['first_traffic_lane_vehicle'] = [first_traffic_lane_vehicle]
                    vehicle['first_group'] = [first_group]

                    # União dos dados
                    df_queue = pd.concat([df_queue,vehicle],ignore_index=True)

                    # Cálculo do novo veículo atrás
                    behind_vehicle = self.FirstVehicleBehind(
                        id_vehicle=id_vehicle,
                        frame=frame,
                        ignore_vehicle_types_list=ignore_vehicle_types_list_distance)
                    behind_vehicle = behind_vehicle[behind_vehicle[self.id_column]!=id_vehicle]
                else:
                    break

            # Atualiza os dados, removendo os veículos das filas já computadas
            df_analysed = df_analysed[-df_analysed[self.id_column].isin(df_queue[self.id_column])].sort_values(by=self.x_head_column,ascending=False)

        df_queue["first_traffic_lane_vehicle"] = df_queue["first_traffic_lane_vehicle"].astype(int)
        
        return df_queue

    def InstantCrossingSection(
        self,
        id_vehicle:int,
        section_reference:float=None,
        logs=False):
        '''
        Calcula o instante de tempo em que o veiculo com o 'id_vehicle' requerido
        cruzou a "section_reference"

        Por problemas de ordem prática, podem ocorrer mais de um cruzamento por
        parte de um mesmo veículo, de modo que considera-se somente o último

        Retorna uma tupla com o primeiro valor sendo o último instante em que a seção
        foi cruzada e o número de vezes que o evento ocorreu.

        -1 ocorre sempre que o veículo não cruzar a seção, portanto o número de
        ocorrências deve ser 0
        '''

        # # Solução simples
        # # Seção padrão da faixa de retenção
        # if section_reference==None:
        #     section_reference = self.motobox_end_section

        # # Dados referentes aos veiculos
        # df_analysed = self.df[self.df[self.id_column]==id_vehicle]

        # # Posição imediatamente antes e depois da seção
        # head_position_after = df_analysed[df_analysed[self.x_head_column]>=section_reference][self.x_head_column].min()
        # head_position_before = df_analysed[df_analysed[self.x_head_column]<section_reference][self.x_head_column].max()
        # dx = head_position_after - head_position_before

        # # Instante em relação às posições escolhidas
        # time_instant_after = df_analysed[df_analysed[self.x_head_column]==head_position_after][self.instant_column].iloc[0]
        # time_instant_before = df_analysed[df_analysed[self.x_head_column]==head_position_before][self.instant_column].iloc[0]
        # dt = time_instant_after - time_instant_before

        # # Instante interpolado para a seção
        # instant_crossing_section = time_instant_before + (dt/dx)*(section_reference-head_position_before)

        # return instant_crossing_section

        # -------------------------------------------------------------------- #
        # # Solução para o problema quando o veículo cruza a seção e recua
        # Seção padrão da faixa de retenção
        if section_reference==None:
            section_reference = self.motobox_end_section

        # Dados referentes aos veiculos
        df_analysed = self.df[self.df[self.id_column]==id_vehicle].sort_values(self.instant_column)

        # Lista de instantes
        instant_list = df_analysed[self.instant_column].tolist()

        # Lista das posições cruzadas
        crossing_section = []
        # Geralmente os veículos começam com "before", mas algumas motos podem iniciar o vídeo com à frente
        # da seção, gerando bugs. Dessa forma, se o veículo estiver à frente, ele receberá "after"
        # Se o veículo não recuar o suficiente para "recruzar" a seção, ele não constará com algum instante
        # antes e depois da seção, recaindo no mesmo if da para a ausência de dados nos últimos veículos
        first_pos = df_analysed[df_analysed[self.instant_column]==min(instant_list)][self.x_head_column].iloc[0]
        status = 'before' if first_pos<section_reference else 'after'
        change_status = {
            'before':'after',
            'after':'before'
        }

        for t in instant_list:
            pos = df_analysed[df_analysed[self.instant_column]==t][self.x_head_column].iloc[0]

            if_status = {
                'before':pos>=section_reference,
                'after':pos<section_reference
            }

            if if_status[status]:
                if status=='before':
                    dx = pos - last_pos
                    dt = t - last_t
                    instant_crossing_section = last_t + (dt/dx)*(section_reference-last_pos)
                    crossing_section.append(instant_crossing_section)
                status = change_status[status]

            last_pos = pos
            last_t = t

        if len(crossing_section)==0:
            print(f'O veículo {id_vehicle} não cruzou a seção') if logs else None
            return -1,0
        else:
            return max(crossing_section),len(crossing_section)

    def VehicleAhead(
        self,
        id_vehicle:int,
        frame:int,
        side_offset_vehicle:float=None,
        max_longitudinal_distance_overlap:float=0.30,
        ignore_vehicle_types_list:list=[],
        project_verification:bool=False
        ):

        # abs(instant_reference_adjust-instant_reference)<0.1
        # Quadro de análise no instante de referencia
        df_analysed = self.df[self.df[self.frame_column]==frame].sort_values(self.frame_column)

        # Veículo particular
        vehicle_df_analysed = df_analysed[df_analysed[self.id_column]==id_vehicle]

        # Se o veículo particular não estiver no instante de tempo, pode-se utilizar a projeção
        if len(vehicle_df_analysed)>0 or project_verification:
            if len(vehicle_df_analysed)==0:
                min_frame_vehicle =  self.df[self.df[self.id_column]==id_vehicle][self.frame_column].min()
                max_frame_vehicle =  self.df[self.df[self.id_column]==id_vehicle][self.frame_column].max()

                frame_vehicle_reference = max_frame_vehicle if frame > max_frame_vehicle else min_frame_vehicle
                vehicle_df_analysed = self.df[(self.df[self.id_column]==id_vehicle) & (self.df[self.frame_column]==frame_vehicle_reference)].sort_values(self.frame_column)
            else:
                pass

            if side_offset_vehicle==None:
                side_offset_vehicle = self.side_offset_vehicle_dict[vehicle_df_analysed[self.vehicle_type_column].iloc[0]]

            side_lim_p1 = vehicle_df_analysed[self.p1_y_bb_column].iloc[0] - side_offset_vehicle
            side_lim_p2 = vehicle_df_analysed[self.p2_y_bb_column].iloc[0] + side_offset_vehicle
            ahead_lim_p2 = vehicle_df_analysed[self.x_head_column].iloc[0] - max_longitudinal_distance_overlap
            behind_lim_p1 = vehicle_df_analysed[self.x_tail_column].iloc[0] + max_longitudinal_distance_overlap

            # Veículo à frente mais próximo
            mask_ahead = (df_analysed[self.p1_y_bb_column]<=side_lim_p2) & (df_analysed[self.p2_y_bb_column]>=side_lim_p1)
            mask_ahead = mask_ahead & (df_analysed[self.x_tail_column]>=ahead_lim_p2) & (-df_analysed[self.vehicle_type_column].isin(ignore_vehicle_types_list))
            ahead_vehicle_group = df_analysed[mask_ahead]
            ahead_vehicle_group['distance_between_vehicles'] = ahead_vehicle_group[self.x_tail_column] - ahead_lim_p2 - max_longitudinal_distance_overlap

        else:
            ahead_vehicle_group = pd.DataFrame(columns=self.df.columns)
            ahead_vehicle_group['distance_between_vehicles'] = []

        return ahead_vehicle_group

    def VehicleBehind(
        self,
        id_vehicle:int,
        frame:int,
        side_offset_vehicle:float=None,
        max_longitudinal_distance_overlap=0.30,
        ignore_vehicle_types_list:list=[]):

        # Quadro de análise no instante de referencia
        df_analysed = self.df[self.df[self.frame_column]==frame].sort_values(self.frame_column)

        # Veículo particular
        vehicle_df_analysed = df_analysed[df_analysed[self.id_column]==id_vehicle]

        if side_offset_vehicle==None:
            side_offset_vehicle = self.side_offset_vehicle_dict[vehicle_df_analysed[self.vehicle_type_column].iloc[0]]

        side_lim_p1 = vehicle_df_analysed[self.p1_y_bb_column].iloc[0] - side_offset_vehicle
        side_lim_p2 = vehicle_df_analysed[self.p2_y_bb_column].iloc[0] + side_offset_vehicle
        ahead_lim_p2 = vehicle_df_analysed[self.x_head_column].iloc[0] - max_longitudinal_distance_overlap
        behind_lim_p1 = vehicle_df_analysed[self.x_tail_column].iloc[0] +  max_longitudinal_distance_overlap

        # Veículo atrás
        mask_behind = (df_analysed[self.p1_y_bb_column]<=side_lim_p2) & (df_analysed[self.p2_y_bb_column]>=side_lim_p1)
        mask_behind = mask_behind & (df_analysed[self.x_head_column]<=behind_lim_p1) & (-df_analysed[self.vehicle_type_column].isin(ignore_vehicle_types_list))
        behind_vehicle_group = df_analysed[mask_behind]
        behind_vehicle_group['distance_between_vehicles'] = behind_lim_p1 - behind_vehicle_group[self.x_head_column] - max_longitudinal_distance_overlap

        return behind_vehicle_group

    def FirstVehicleAhead(
        self,
        id_vehicle:int,
        frame:int,
        side_offset_vehicle:float=None,
        max_longitudinal_distance_overlap:float=0.30,
        ignore_vehicle_types_list:list=[],
        project_verification:bool=False,
        report_null_value:bool=False):

        ahead_vehicle_group = self.VehicleAhead(
            id_vehicle=id_vehicle,
            frame=frame,
            side_offset_vehicle=side_offset_vehicle,
            max_longitudinal_distance_overlap=max_longitudinal_distance_overlap,
            ignore_vehicle_types_list=ignore_vehicle_types_list,
             project_verification=project_verification
        )

        pos_ahead_vehicle = ahead_vehicle_group['distance_between_vehicles'].min()
        ahead_vehicle = ahead_vehicle_group[ahead_vehicle_group['distance_between_vehicles']==pos_ahead_vehicle]

        if report_null_value and len(ahead_vehicle)==0:
            ahead_vehicle[self.id_column] = [-1]
        
        # Para evitar erros com veículos "dupicados"
        if len(ahead_vehicle)>1:
            ahead_vehicle = ahead_vehicle.iloc[:1]

        return ahead_vehicle

    def FirstVehicleBehind(
        self,
        id_vehicle:int,
        frame:int,
        side_offset_vehicle:float=None,
        max_longitudinal_distance_overlap:float=0.30,
        ignore_vehicle_types_list:list=[]):

        behind_vehicle_group = self.VehicleBehind(
            id_vehicle=id_vehicle,
            frame=frame,
            side_offset_vehicle=side_offset_vehicle,
            max_longitudinal_distance_overlap=max_longitudinal_distance_overlap,
            ignore_vehicle_types_list=ignore_vehicle_types_list
        )

        pos_behind_vehicle = behind_vehicle_group['distance_between_vehicles'].min()
        behind_vehicle = behind_vehicle_group[behind_vehicle_group['distance_between_vehicles']==pos_behind_vehicle]

        # Para evitar erros com veículos "dupicados"
        if len(behind_vehicle)>1:
            behind_vehicle = behind_vehicle.iloc[:1]

        return behind_vehicle

    def VehicleMotoboxDetect(self,frame,transverse_offset_motobox:float=0,longitudinal_offset_motobox:float=0):
        '''
        Análise baseada no centríde da motocicleta
        '''
        # Ajuste do instante de tempo
        # instant_reference = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]

        # Quadro de análise no instante de referencia
        df_analysed = self.df[self.df[self.frame_column]==frame].sort_values(self.frame)

        if df_analysed.empty:
            return df_analysed

        # Limites transversais
        up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[0])(x) - transverse_offset_motobox
        down_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[-1])(x) + transverse_offset_motobox
        df_analysed = df_analysed[df_analysed[self.y_centroid_column] >= df_analysed[self.x_centroid_column].apply(lambda x:up_lim_func(x))]
        df_analysed = df_analysed[df_analysed[self.y_centroid_column] <= df_analysed[self.x_centroid_column].apply(lambda x:down_lim_func(x))]

        # Limites longitudinais
        start_lim = self.motobox_start_section - longitudinal_offset_motobox
        end_lim = self.motobox_end_section + longitudinal_offset_motobox
        df_analysed = df_analysed[df_analysed[self.x_centroid_column] <= end_lim]
        df_analysed = df_analysed[df_analysed[self.x_centroid_column] >= start_lim]

        return df_analysed

    def PolygonalVeicleVirtualLane(
        self,
        instant_reference:float,
        transverse_offset_motobox:float=None,
        longitudinal_offset_motobox:float=0,
        include_motorcycle_box=True,
        max_transversel_distance_overlap=0
        ):

        '''
        OBSOLETO
        '''

        # raise ValueError('Função OBSOLETA')

        if transverse_offset_motobox==None:
            transverse_offset_motobox = 0.5*self.width_virtual_lane

        # Ajuste do instante de tempo
        instant_reference = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]

        # Quadro de análise no instante de referencia
        df_analysed = self.df[self.df[self.instant_column]==instant_reference]

        # Motobox
        if not include_motorcycle_box:
            # Veículos no motobox
            vehicle_motorcycle_box = self.VehicleMotoboxDetect(
                instant_reference=instant_reference,
                transverse_offset_motobox=transverse_offset_motobox,
                longitudinal_offset_motobox=longitudinal_offset_motobox
            )

        # Quantidade de corredores virtuais (centro e laterais)
        number_virtual_lane = len(self.df[self.traffic_lane_column].unique()) + 1
        virtual_lane_list = [i for i in range(1,number_virtual_lane+1)]

        # Dataset principal
        vehicle_virtual_lane = pd.DataFrame()

        # Cada instante tem um corredor virtual que é função dos limites laterais dos veiculos, geometria e
        # limites aritrarios quando não houver veiculos
        # ------------------------------------| VL1
        #         ________________            |
        #         |               |           | L1
        #         |_______________|           |
        # ---------[X-c]---------[X-c]--------| VL2
        #                 ________________    |
        #                 |               |   | L2
        #                 |_______________|   |
        # ------------------------------------| VL3
        #                                     |
        #                                     | L3
        #                                     |
        # ------------------------------------| VL4

        for virtual_lane in virtual_lane_list:
            # Definição dos veículos que formam os limites das faixas
            # fwv - veiclos de 4 rodas naquele instante t
            four_wheel_vehicle = df_analysed[df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['four_wheel'])].sort_values(self.x_centroid_column)

            # Faixa 1
            if virtual_lane==1:
                fwv_on_below_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                if len(fwv_on_below_lane)>0:
                    inf_lim_x = []
                    inf_lim_y = []
                    for i in range(len(fwv_on_below_lane)):
                        inf_lim_x.append(fwv_on_below_lane[self.p1_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])
                        inf_lim_x.append(fwv_on_below_lane[self.p2_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])

                    inf_lim = pd.DataFrame()
                    inf_lim['x'] = inf_lim_x
                    inf_lim['y'] = inf_lim_y
                    inf_lim = inf_lim.sort_values('x').values.tolist()
                    inf_lim_func = PolygonalToFunction(inf_lim)
                else:
                    inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Ultima faixa
            elif virtual_lane==max(virtual_lane_list):
                fwv_on_above_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane-1].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                if len(fwv_on_above_lane)>0:
                    up_lim_x = []
                    up_lim_y = []

                    for i in range(len(fwv_on_above_lane)):
                        up_lim_x.append(fwv_on_above_lane[self.p1_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])
                        up_lim_x.append(fwv_on_above_lane[self.p2_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])

                    up_lim = pd.DataFrame()
                    up_lim['x'] = up_lim_x
                    up_lim['y'] = up_lim_y
                    up_lim = up_lim.sort_values('x').values.tolist()
                    up_lim_func = PolygonalToFunction(up_lim)
                else:
                    up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Demais faixas
            else:
                fwv_on_above_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane-1].sort_values(self.x_centroid_column)
                fwv_on_below_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                if len(fwv_on_above_lane)>0:
                    up_lim_x = []
                    up_lim_y = []

                    for i in range(len(fwv_on_above_lane)):
                        up_lim_x.append(fwv_on_above_lane[self.p1_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])
                        up_lim_x.append(fwv_on_above_lane[self.p2_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])

                    up_lim = pd.DataFrame()
                    up_lim['x'] = up_lim_x
                    up_lim['y'] = up_lim_y
                    up_lim = up_lim.sort_values('x').values.tolist()
                    up_lim_func = PolygonalToFunction(up_lim)
                else:
                    up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                if len(fwv_on_below_lane)>0:
                    inf_lim_x = []
                    inf_lim_y = []
                    for i in range(len(fwv_on_below_lane)):
                        inf_lim_x.append(fwv_on_below_lane[self.p1_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])
                        inf_lim_x.append(fwv_on_below_lane[self.p2_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])

                    inf_lim = pd.DataFrame()
                    inf_lim['x'] = inf_lim_x
                    inf_lim['y'] = inf_lim_y
                    inf_lim = inf_lim.sort_values('x').values.tolist()
                    inf_lim_func = PolygonalToFunction(inf_lim)
                else:
                    inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Veiculos dentro do limite
            vehicle_on_virtual_lane = df_analysed[df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['two_wheel'])]
            # Remove motocicletas no motobox
            if not include_motorcycle_box:
                vehicle_on_virtual_lane = vehicle_on_virtual_lane[-vehicle_on_virtual_lane[self.id_column].isin(vehicle_motorcycle_box[self.id_column])]

            # Notas:
            # O parâmetro "max_transversel_distance_overlap" determina um aumento dos limites das faixas em relação
            # a lateral dos veículos que determinam o corredor virtual
            # É uma solução temporária para considerar motos que estão no corredor de uma faixa
            # mas em relação à outra faixa não estão "distantes" o suficiente
            # Para ser mais rigoroso, zerar o parâmetro
            # vehicle_on_virtual_lane['up_lim'] = vehicle_on_virtual_lane[self.x_centroid_column].apply(lambda x:up_lim_func(x))
            vehicle_on_virtual_lane['up_lim'] = vehicle_on_virtual_lane.apply(lambda x:up_lim_func(x[self.x_centroid_column])-max_transversel_distance_overlap,axis=1)
            # vehicle_on_virtual_lane['inf_lim'] = vehicle_on_virtual_lane[self.x_centroid_column].apply(lambda x:inf_lim_func(x))
            vehicle_on_virtual_lane['inf_lim'] = vehicle_on_virtual_lane.apply(lambda x:inf_lim_func(x[self.x_centroid_column])+max_transversel_distance_overlap,axis=1)

            vehicle_on_virtual_lane = vehicle_on_virtual_lane[(vehicle_on_virtual_lane['up_lim'] <= vehicle_on_virtual_lane[self.y_centroid_column]-0.25*vehicle_on_virtual_lane[self.vehicle_width_column]) & (vehicle_on_virtual_lane['inf_lim'] >= vehicle_on_virtual_lane[self.y_centroid_column]+0.25*vehicle_on_virtual_lane[self.vehicle_width_column])]
            vehicle_on_virtual_lane[self.virtual_traffic_lane_column] = -1

            vehicle_virtual_lane = pd.concat([vehicle_virtual_lane,vehicle_on_virtual_lane],ignore_index=True)

            vehicle_virtual_lane.loc[vehicle_virtual_lane[self.id_column].isin(vehicle_on_virtual_lane[self.id_column].tolist()),self.virtual_traffic_lane_column] = virtual_lane
            vehicle_virtual_lane = vehicle_virtual_lane.sort_values(by=[self.virtual_traffic_lane_column,self.x_centroid_column])

        return vehicle_virtual_lane

    def MotorcycleInVirtualLane(
        self,
        frame,
        lat_virtual_lane_overlap:float=0,
        include_motorcycle_box=True,
        lon_offset_motobox:float=0,
        lat_offset_motobox:float=None,
        ):

        '''
        Susbtitui a função "PolygonalVeicleVirtualLane"
        '''
        # Ajusta caixa de referência do que é definido no motobox
        if lat_offset_motobox==None:
            lat_offset_motobox = 0.5*self.width_virtual_lane

        # Frame de análise no instante de referencia
        df_analysed = self.df[self.df[self.frame_column]==frame]

        if df_analysed.empty:
            return df_analysed

        # Motobox
        if not include_motorcycle_box:
            # Veículos no motobox
            vehicle_motorcycle_box = self.VehicleMotoboxDetect(
                frame=frame,
                transverse_offset_motobox=lat_offset_motobox,
                longitudinal_offset_motobox=lon_offset_motobox
            )

        # Quantidade de corredores virtuais (centro e laterais)
        number_virtual_lane = self.traffic_lane_polygon["id"].astype(int).max()
        virtual_lane_list = [i for i in range(1,number_virtual_lane+1)]

        # Dataset principal
        vehicle_virtual_lane = pd.DataFrame()

        # Cada instante tem um corredor virtual que é função dos limites laterais dos veiculos, geometria e
        # limites aritrarios quando não houver veiculos
        # ------------------------------------| VL1
        #         ________________            |
        #         |               |           | L1
        #         |_______________|           |
        # ---------[X-c]---------[X-c]--------| VL2
        #                 ________________    |
        #                 |               |   | L2
        #                 |_______________|   |
        # ------------------------------------| VL3
        #                                     |
        #                                     | L3
        #                                     |
        # ------------------------------------| VL4

        # Definição das funções de contorno de cada faixa
        for virtual_lane in virtual_lane_list:
            # Definição dos veículos que formam os limites das faixas
            # fwv - veiclos de 4 rodas naquele instante t
            four_wheel_vehicle = df_analysed[df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['four_wheel'])].sort_values(self.x_centroid_column)

            # Faixa 1
            if virtual_lane==1:
                fwv_on_below_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                if len(fwv_on_below_lane)>0:
                    inf_lim_x = []
                    inf_lim_y = []
                    for i in range(len(fwv_on_below_lane)):
                        inf_lim_x.append(fwv_on_below_lane[self.p1_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])
                        inf_lim_x.append(fwv_on_below_lane[self.p2_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])

                    inf_lim = pd.DataFrame()
                    inf_lim['x'] = inf_lim_x
                    inf_lim['y'] = inf_lim_y
                    inf_lim = inf_lim.sort_values('x').values.tolist()
                    inf_lim_func = PolygonalToFunction(inf_lim)
                else:
                    inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Ultima faixa
            elif virtual_lane==max(virtual_lane_list):
                fwv_on_above_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane-1].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                if len(fwv_on_above_lane)>0:
                    up_lim_x = []
                    up_lim_y = []

                    for i in range(len(fwv_on_above_lane)):
                        up_lim_x.append(fwv_on_above_lane[self.p1_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])
                        up_lim_x.append(fwv_on_above_lane[self.p2_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])

                    up_lim = pd.DataFrame()
                    up_lim['x'] = up_lim_x
                    up_lim['y'] = up_lim_y
                    up_lim = up_lim.sort_values('x').values.tolist()
                    up_lim_func = PolygonalToFunction(up_lim)
                else:
                    up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Demais faixas
            else:
                fwv_on_above_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane-1].sort_values(self.x_centroid_column)
                fwv_on_below_lane = four_wheel_vehicle[four_wheel_vehicle[self.traffic_lane_column]==virtual_lane].sort_values(self.x_centroid_column)

                # Definição dos limites
                # Superior
                if len(fwv_on_above_lane)>0:
                    up_lim_x = []
                    up_lim_y = []

                    for i in range(len(fwv_on_above_lane)):
                        up_lim_x.append(fwv_on_above_lane[self.p1_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])
                        up_lim_x.append(fwv_on_above_lane[self.p2_x_bb_column].iloc[i])
                        up_lim_y.append(fwv_on_above_lane[self.p2_y_bb_column].iloc[i])

                    up_lim = pd.DataFrame()
                    up_lim['x'] = up_lim_x
                    up_lim['y'] = up_lim_y
                    up_lim = up_lim.sort_values('x').values.tolist()
                    up_lim_func = PolygonalToFunction(up_lim)
                else:
                    up_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) - 0.5*self.width_virtual_lane

                # Inferior
                if len(fwv_on_below_lane)>0:
                    inf_lim_x = []
                    inf_lim_y = []
                    for i in range(len(fwv_on_below_lane)):
                        inf_lim_x.append(fwv_on_below_lane[self.p1_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])
                        inf_lim_x.append(fwv_on_below_lane[self.p2_x_bb_column].iloc[i])
                        inf_lim_y.append(fwv_on_below_lane[self.p1_y_bb_column].iloc[i])

                    inf_lim = pd.DataFrame()
                    inf_lim['x'] = inf_lim_x
                    inf_lim['y'] = inf_lim_y
                    inf_lim = inf_lim.sort_values('x').values.tolist()
                    inf_lim_func = PolygonalToFunction(inf_lim)
                else:
                    inf_lim_func = lambda x:PointToFunction(self.virtual_lane_lim[virtual_lane-1])(x) + 0.5*self.width_virtual_lane

            # Veiculos dentro do limite
            vehicle_on_virtual_lane = df_analysed[df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['two_wheel'])]
            # Remove motocicletas no motobox, se for o caso
            if not include_motorcycle_box:
                vehicle_on_virtual_lane = vehicle_on_virtual_lane[-vehicle_on_virtual_lane[self.id_column].isin(vehicle_motorcycle_box[self.id_column])]

            # Notas:
            # O parâmetro "lat_virtual_lane_overlap" determina um aumento dos limites das faixas em relação
            # a lateral dos veículos que determinam o corredor virtual
            # É uma solução temporária para considerar motos que estão no corredor de uma faixa
            # mas em relação à outra faixa não estão "distantes" o suficiente
            # Para ser mais rigoroso, zerar o parâmetro
            # vehicle_on_virtual_lane['up_lim'] = vehicle_on_virtual_lane[self.x_centroid_column].apply(lambda x:up_lim_func(x))
            vehicle_on_virtual_lane['up_lim'] = vehicle_on_virtual_lane.apply(lambda x:up_lim_func(x[self.x_centroid_column])-lat_virtual_lane_overlap,axis=1)
            # vehicle_on_virtual_lane['inf_lim'] = vehicle_on_virtual_lane[self.x_centroid_column].apply(lambda x:inf_lim_func(x))
            vehicle_on_virtual_lane['inf_lim'] = vehicle_on_virtual_lane.apply(lambda x:inf_lim_func(x[self.x_centroid_column])+lat_virtual_lane_overlap,axis=1)

            vehicle_on_virtual_lane = vehicle_on_virtual_lane[(vehicle_on_virtual_lane['up_lim'] <= vehicle_on_virtual_lane[self.y_centroid_column]-0.5*vehicle_on_virtual_lane[self.vehicle_width_column]) & (vehicle_on_virtual_lane['inf_lim'] >= vehicle_on_virtual_lane[self.y_centroid_column]+0.50*vehicle_on_virtual_lane[self.vehicle_width_column])]
            vehicle_on_virtual_lane[self.virtual_traffic_lane_column] = -1

            vehicle_virtual_lane = pd.concat([vehicle_virtual_lane,vehicle_on_virtual_lane],ignore_index=True)

            vehicle_virtual_lane.loc[vehicle_virtual_lane[self.id_column].isin(vehicle_on_virtual_lane[self.id_column].tolist()),self.virtual_traffic_lane_column] = virtual_lane
            vehicle_virtual_lane = vehicle_virtual_lane.sort_values(by=[self.virtual_traffic_lane_column,self.x_centroid_column])

        return vehicle_virtual_lane

    # OBSOLETO
    def Tp1MotorcycleAnalysis2(
        self,
        instant_reference:float,
        N:int=4,
        distance_between_motorcycle_and_vehicle_ahead_1:float=3.0,
        distance_between_motorcycle_and_vehicle_ahead_2:float=4.5,
        distance_between_motorcycle_and_motorcycle_ahead:float=1.5,
        distance_between_motorcycle_ahead_virtual_lane:float=1.0,
        distance_between_motorcycle_behind_virtual_lane:float=2.5,
        motobox_max_invasion_distance:float=1,
        max_transversel_distance_overlap_virtual_lane:float=0.30,
        max_longitudinal_distance_overlap:float=0.30,
        ghost_frames_generator_range=30,
        logs=True,
    ):
        '''
        Parâmetros padrão (unidade em metros)
        # Distancias a frente do primeiro veículo
        - Limites para o caso 1
        distance_between_motorcycle_and_vehicle_ahead_1 = 2.5
        - Limite para o caso 2
        distance_between_motorcycle_and_vehicle_ahead_2 = 4
        - Limites para o caso 2 e 3
        distance_between_motorcycle_and_motorcycle_ahead = 1

        # Distância longitudinal dos veiculos no corredor virtual
        - Limite a frente do veículo
        distance_between_motorcycle_ahead_virtual_lane = 0.5
        - Limite atrás da frente do veículo (campo de visão)
        distance_between_motorcycle_behind_virtual_lane = 2.5

        # Limite de invasão dos motobox pelos veículos
        motobox_max_invasion_distance = 1

        A função retorna um pd.DataFrame com os dados padronizados para o tp1
        '''
        # Futuramente pode ser alocada em outro canto isso
        # Geração de informações de frames possívelmente ausentes
        # Frame de referencia
        # Ajuste do instante de tempo
        instant_reference = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]
        frame_reference = self.df[self.df[self.instant_column]==instant_reference][self.frame_column].iloc[0]
        inf_frame = frame_reference-int(ghost_frames_generator_range/2)
        sup_frame = frame_reference+int(ghost_frames_generator_range/2)
        id_vehicle_list = self.df[self.df[self.instant_column].between(inf_frame,sup_frame)][self.id_column].tolist()

        # Geração de frames
        generated_ghost_frames = self.GhostFramesGenerator(id_vehicle_list,range_frame=(inf_frame,sup_frame))
        # Mesclar com df
        self.df = pd.concat([self.df,generated_ghost_frames],ignore_index=True).sort_values(by=[self.frame_column,self.id_column])

        # Inicio da função propriamente dita
        # Toma as filas formadas no ciclo
        df_queue = self.QueueDetector(instant_reference,distance_between_vehicles_lim=1e10)
        # Toma a primeira fila, mais proxima da faixa de retenção
        df_queue = df_queue[df_queue['first_group']==1]
        # Faixas
        traffic_lane_list = df_queue.sort_values(by=[self.traffic_lane_column,self.queue_position_column])[self.traffic_lane_column].unique().tolist()

        # Dados do tp1
        df_tp1 = pd.DataFrame()

        for traffic_lane in traffic_lane_list:
            valid_lane = True
            report = ''
            # Fila de veículos em uma determinada faixa
            df_analysed = df_queue[df_queue[self.traffic_lane_column]==traffic_lane]
            instant_min = str(int(instant_reference/60))+':'+(str(int(round(instant_reference % 60,0))) if len(str(int(round(instant_reference % 60,0))))>1 else '0'+str(int(round(instant_reference % 60,0))))

            # Exceções
            # Se houver a invasão de veículos de 4 rodas no motobox
            vehicle_motobox_invasion = df_analysed[df_analysed[self.x_head_column]>=self.motobox_start_section+motobox_max_invasion_distance]
            if len(vehicle_motobox_invasion)>0:
                report = report + '@' if len(report)>0 else ''
                report = report + f'O(s) veículo(s) {vehicle_motobox_invasion[self.id_column].tolist()} estão invadindo o motobox no instante {instant_min}'
                valid_lane = False

            # self.side_offset_vehicle_dict
            # Se nãou houver pelo menos N veículos na fila
            if len(df_analysed)<N:
                report = report + '@' if len(report)>0 else ''
                report = report + f'Tamanho de fila no instante {instant_min} na faixa {traffic_lane} é menor que {N}'
                valid_lane = False
            else:
                # --------------------------------------------------------------
                # Variável a ser explicada - headway acumulado do 4o veiculo
                idN = df_analysed[df_analysed[self.queue_position_column]==N].iloc[0][self.id_column]
                instant_crossing,instant_crossing_log = self.InstantCrossingSection(idN)
                H4j = instant_crossing - instant_reference

                # Se o headway for negativo, provavelmente o vídeo acabou antes do veículo cruzar a seção
                if H4j<0:
                    report = report + '@' if len(report)>0 else ''
                    report = report + f'O veículo {idN} não cruzou a faixa de interesse'
                    valid_lane = False

                # Verifica a distância entre cada veículo até a N posição
                # Calcula a fila real entre os veículos, coonsiderando motos entre veículos
                # Se alguma das distâncias calculadas for maior que 5m, não considera fila
                # Se alguma das distâncias for maior que 5m, verifica uma a uma, caso a
                # maior que 5m  não possua moto entre os veículos, exclui o ciclo
                pos_count = 1
                new_id = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.id_column]
                vehicle_type = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.vehicle_type_column]
                distance_to_vehicle_behind = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0]['distance_between_vehicles']
                # Veículo restritamente atrás
                vehicle_behind = self.FirstVehicleBehind(
                    new_id,
                    instant_reference,
                    side_offset_vehicle=self.side_offset_vehicle_dict[vehicle_type]*2
                )
                while pos_count<len(df_analysed):
                    if (distance_to_vehicle_behind<8) or (vehicle_behind.iloc[0]['distance_between_vehicles']<8):
                        pos_count = pos_count + 1
                        new_id = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.id_column]
                        vehicle_type = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.vehicle_type_column]
                        distance_to_vehicle_behind = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0]['distance_between_vehicles']
                        # Veículo restritamente atrás
                        vehicle_behind = self.FirstVehicleBehind(
                            new_id,
                            instant_reference,
                            side_offset_vehicle=self.side_offset_vehicle_dict[vehicle_type]*2
                        )
                    else:
                        report = report + '@' if len(report)>0 else ''
                        report = report + f'Tamanho de fila no instante {instant_min} na faixa {traffic_lane} é menor que {N}, pois o veículo {new_id} está muito distante'
                        valid_lane = False
                        break

            if valid_lane:
                # --------------------------------------------------------------
                # Variável explicativa 1
                # Motcicletas à frente do primeiro veículo
                # Id do primeiro veiculo
                id1 = df_analysed[df_analysed[self.queue_position_column]==1].iloc[0][self.id_column]
                # Todas as motocicletas à frente do priemiro veículo, incluindo lateralizadas
                all_motorcycles = self.VehicleAhead(id1,instant_reference,side_offset_vehicle=0.30).sort_values(by='distance_between_vehicles')
                motorcycles_ahead_projection = self.VehicleAhead(
                    id1,
                    instant_reference,
                    max_longitudinal_distance_overlap=max_longitudinal_distance_overlap
                    ).sort_values(by='distance_between_vehicles')

                # Todas as motocicletas
                idQmfj = all_motorcycles[self.id_column].tolist()
                motocycle_group_by_class = {
                    '10':[],
                    '11':[],
                    '20':[],
                    '21':[],
                    '30':[],
                    '31':[],
                }

                for id_motocycle in idQmfj:
                    group_motocycle_ahead = ''
                    # Motocicleta
                    vehicle = all_motorcycles[all_motorcycles[self.id_column]==id_motocycle]

                    if vehicle['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_vehicle_ahead_1:
                        group_motocycle_ahead = group_motocycle_ahead + '1'
                    elif vehicle['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_vehicle_ahead_2:
                        group_motocycle_ahead = group_motocycle_ahead + '2'
                    else:
                        group_motocycle_ahead = group_motocycle_ahead + '3'

                    motocycle_ahead = self.FirstVehicleAhead(id_motocycle,instant_reference,max_longitudinal_distance_overlap=max_longitudinal_distance_overlap)
                    # motocycle_ahead = all_motorcycles[]

                    if len(motocycle_ahead)>0:
                        if motocycle_ahead['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_motorcycle_ahead:
                            group_motocycle_ahead = group_motocycle_ahead + '1'
                        else:
                            group_motocycle_ahead = group_motocycle_ahead + '0'
                    else:
                        group_motocycle_ahead = group_motocycle_ahead + '0'

                    motocycle_group_by_class[group_motocycle_ahead].append(id_motocycle)

                idQmf10j = motocycle_group_by_class['10']
                idQmf11j = motocycle_group_by_class['11']
                idQmf20j = motocycle_group_by_class['20']
                idQmf21j = motocycle_group_by_class['21']
                idQmf30j = motocycle_group_by_class['30']
                idQmf31j = motocycle_group_by_class['31']

                idQmfj = idQmf10j+idQmf11j+idQmf20j+idQmf21j+idQmf30j+idQmf31j

                # --------------------------------------------------------------
                # Variável explicativa 2 - motos entre veiculos
                # Veículos da posição 2 a 4
                idQmevj = []

                for pos in range(2,N+1):
                    # # Posição correspondente a frente do carro da posição "pos"
                    # lim_distance_between_motorcycle_behind = df_analysed[df_analysed[self.queue_position_column]==pos][self.x_head_column].iloc[0]

                    # # Veíuclo na posição "pos - 1"
                    # vehicle = df_analysed[df_analysed[self.queue_position_column]==pos-1]

                    # # Motos atrás do veículo e com a traseira da moto à frente da posição da frente do veículo de trás
                    # motocycle_behind = self.VehicleBehind(vehicle[self.id_column].iloc[0],instant_reference,side_offset_vehicle=0.30,ignore_vehicle_types_list=self.vehicle_category_list['four_wheel'])

                    # motocycle_behind = motocycle_behind[motocycle_behind[self.x_tail_column]>=lim_distance_between_motorcycle_behind]

                    # idQmevjp = motocycle_behind[self.id_column].tolist()

                    # idQmevj.append(idQmevjp)

                    # Posição correspondente a atrás do carro da posição "pos-1"
                    lim_distance_between_motorcycle_ahead = df_analysed[df_analysed[self.queue_position_column]==pos-1][self.x_tail_column].iloc[0]
                    lim_distance_between_motorcycle_ahead = lim_distance_between_motorcycle_ahead + max_longitudinal_distance_overlap

                    # Veíuclo na posição "pos"
                    vehicle = df_analysed[df_analysed[self.queue_position_column]==pos]

                    # Motos a frnte do veículo e com a traseira da moto à frente da posição da frente do veículo de trás
                    # Mesmos critérios para a variável moto a frente do 1º veículo
                    motocycle_ahead = self.VehicleAhead(
                        vehicle[self.id_column].iloc[0],
                        instant_reference,
                        side_offset_vehicle=0.30,
                        max_longitudinal_distance_overlap=max_longitudinal_distance_overlap,
                        ignore_vehicle_types_list=self.vehicle_category_list['four_wheel'])

                    # Filtra as motos longitudinalmente
                    motocycle_ahead = motocycle_ahead[motocycle_ahead[self.x_head_column]<=lim_distance_between_motorcycle_ahead]

                    # Coleta o id
                    idQmevjp = motocycle_ahead[self.id_column].tolist()
                    idQmevj.append(idQmevjp)

                # --------------------------------------------------------------
                # Variável explicativa 3 - motos no corredor virtual
                # Limites longitudinais
                lim_first_vehicle = df_analysed[df_analysed[self.queue_position_column]==1].iloc[0][self.x_head_column] + distance_between_motorcycle_ahead_virtual_lane
                lim_N_vehicle = df_analysed[df_analysed[self.queue_position_column]==N].iloc[0][self.x_head_column] - distance_between_motorcycle_behind_virtual_lane
                # Todas as motos do correor virtual
                # motocycle_virtural_lane = self.PolygonalVeicleVirtualLane(
                #     instant_reference,
                #     max_transversel_distance_overlap=max_transversel_distance_overlap_virtual_lane)

                motocycle_virtural_lane = self.MotorcycleInVirtualLane(
                    instant_reference=instant_reference,
                    lat_virtual_lane_overlap=0.8,
                    include_motorcycle_box=True)

                # Filtra quais corredores são contabilizados
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.virtual_traffic_lane_column].isin([traffic_lane,traffic_lane+1])]

                # Filtra as motos nos limites longitudinais
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.x_tail_column]<=lim_first_vehicle]
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.x_head_column]>=lim_N_vehicle]

                # Ids das motos até o limite lateral das motos
                motorcycle_side = pd.concat([self.SideVehicle(id_vehicle,instant_reference,overlap_lat=max_transversel_distance_overlap_virtual_lane) for id_vehicle in df_analysed[df_analysed[self.queue_position_column].between(1,N)][self.id_column].tolist()])
                motorcycle_side = motorcycle_side[motorcycle_side['lateral_distance_between_vehicles'].abs()<=1][self.id_column].unique()
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.id_column].isin(motorcycle_side)]

                # Ids das motos no corredor virtual
                # Densconsidera veículos previamente alocados entre veículos, e a frente, caso já exista
                idQmcvj = [i for i in motocycle_virtural_lane[self.id_column].tolist() if i not in JoinList(idQmevj)+idQmfj]


                # --------------------------------------------------------------
                # Variável explicativa 4 - Motos efetivamente impactando
                idQmcvpj = []
                for pos in range(1,N+1):
                    # Veiculo
                    vehicle = df_analysed[df_analysed[self.queue_position_column]==pos]

                    # Limites longitudinais
                    lim_ahead = vehicle.iloc[0][self.x_head_column] + distance_between_motorcycle_ahead_virtual_lane
                    lim_behind = vehicle.iloc[0][self.x_head_column] - distance_between_motorcycle_behind_virtual_lane

                    # Filtra as motos nos limites longitudinais
                    motocycle_virtural_lane_pos = motocycle_virtural_lane[motocycle_virtural_lane[self.x_tail_column]<=lim_ahead]
                    motocycle_virtural_lane_pos = motocycle_virtural_lane_pos[motocycle_virtural_lane_pos[self.x_head_column]>=lim_behind]

                    idQmcvpj.append(motocycle_virtural_lane_pos[self.id_column].tolist())


                idQmcvpj = [[j for j in i if j not in JoinList(idQmevj)+idQmfj] for i in idQmcvpj]

                # --------------------------------------------------------------
                # Variável explicativa 5 - Densidade de motocicletas no corredor virtual
                # Densidade do corredor virtual (2 corredores, cada um de cada lado)
                Dcvj = len(idQmcvj)/(2*(lim_first_vehicle-lim_N_vehicle))

                # --------------------------------------------------------------
                # Variável 6 - Quantidade de veículos pesados
                heavy_vehicle = df_analysed[(df_analysed[self.queue_position_column].between(1,N)) & (df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['heavy']))]
                idQvpj = heavy_vehicle[self.id_column].tolist()

                # --------------------------------------------------------------
                # Variável 7 - Motobox
                if self.motobox_start_section==self.motobox_end_section:
                    MB = 0
                else:
                    MB = 1

                summary_tp1 = {
                    'report':[report],
                    'frame':[frame_reference],
                    'instant':[instant_min],
                    'traffic_lane':[traffic_lane],
                    'idN':[idN],
                    'H4j':[round(H4j,2)],

                    'idQmevj':[idQmevj],
                    'Qmevj':[sum([len(i) for i in idQmevj])],

                    'idQmfj':[idQmfj],
                    'Qmfj':[len(idQmfj)],
                    'idQmf10j':[idQmf10j],
                    'Qmf10j':[len(idQmf10j)],
                    'idQmf20j':[idQmf20j],
                    'Qmf20j':[len(idQmf20j)],
                    'idQmf30j':[idQmf30j],
                    'Qmf30j':[len(idQmf30j)],
                    'idQmf11j':[idQmf11j],
                    'Qmf11j':[len(idQmf11j)],
                    'idQmf21j':[idQmf21j],
                    'Qmf21j':[len(idQmf21j)],
                    'idQmf31j':[idQmf31j],
                    'Qmf31j':[len(idQmf31j)],

                    'idQmcvj':[idQmcvj],
                    'Qmcvj':[len(idQmcvj)],

                    'idQmcvpj':[idQmcvpj],
                    'Qmcvpj':[sum([len(i) for i in idQmcvpj])],

                    'Dcvj':[round(Dcvj,2)],

                    'idQvpj':[idQvpj],
                    'Qvpj':[len(idQvpj)],

                    'MB':[MB]
                }

                df_tp1 = pd.concat([df_tp1,pd.DataFrame().from_dict(summary_tp1)],ignore_index=True)

            else:
                print(report) if logs else None
                summary_tp1 = {
                    'report':[report],
                    'frame':[frame_reference],
                    'instant':[instant_min],
                    'traffic_lane':[traffic_lane],
                    'idN':[[]],
                    'H4j':[0],

                    'idQmevj':[[]],
                    'Qmevj':[0],

                    'idQmfj':[[]],
                    'Qmfj':[0],
                    'idQmf10j':[[]],
                    'Qmf10j':[0],
                    'idQmf20j':[[]],
                    'Qmf20j':[0],
                    'idQmf30j':[[]],
                    'Qmf30j':[0],
                    'idQmf11j':[[]],
                    'Qmf11j':[0],
                    'idQmf21j':[[]],
                    'Qmf21j':[0],
                    'idQmf31j':[[]],
                    'Qmf31j':[0],

                    'idQmcvj':[[]],
                    'Qmcvj':[0],

                    'idQmcvpj':[0],
                    'Qmcvpj':[0],

                    'Dcvj':[0],

                    'idQvpj':[[]],
                    'Qvpj':[0],

                    'MB':[0]
                }

                df_tp1 = pd.concat([df_tp1,pd.DataFrame().from_dict(summary_tp1)],ignore_index=True)

        return df_tp1
    
    def FirstVehicleQueue(
            self,
            frame,
            traffic_lane,
            distance_between_vehicles_lim=1e10):
        """
        Retorna o id do primeiro veículo em uma faixa especificada em uma fila "ilimitada"
        Se não tiver veículo, retorna None
        """
        # Veículos em fila na faixa especificada
        df_queue = self.QueueDetector(
            frame=frame,
            distance_between_vehicles_lim=distance_between_vehicles_lim)
        
        # Verifica se retornou um dataframe vazio, retorna np.nan
        if df_queue.empty:
            return np.nan
        
        # Filtra o primeiro veículo
        query_exp = f"queue_position == 1 & first_traffic_lane_vehicle == {traffic_lane}"
        df_queue = df_queue.query(query_exp)

        # Verifica se retornou um dataframe vazio, retorna np.nan
        if df_queue.empty:
            return np.nan
        
        # Id do veiculo na primeira posição
        id_1st = df_queue[self.id_column].values[0]
        
        return id_1st

    def TimeSpaceHeadway(
            self,
            vehicle_id,
            frame,
            section,
            x_column=None,
            y_column=None,
            ):
        """
        Calcula o headway do primeiro veículo em relação a uma seção de referência
        Inclui variáveis para a modelagem do hd1
        """
        # Headway temporal do primeiro veículo em segundos
        hd1_time = 0
        # Headway espacial do primeiro veículo em metros
        hd1_space = 0

        # Ajusta as colunas padrão
        if x_column==None:
            x_column = self.x_head_column
        if y_column==None:
            y_column = self.y_centroid_column

        query_expr = f"{self.id_column} == {vehicle_id} & {self.frame_column} >= {frame}"
        df_traj = self.df.query(query_expr).sort_values(self.frame_column)

        # Verifica se o frame passado existe na trajetória do veículo (deve ser a primeira)
        # Se não for, pode existir trajetória após o frame coletado, mas o headway tem outra
        # interpretação
        if not df_traj[self.frame_column].values[0]==frame:
            raise ValueError(f"O veículo não presenta registro no frame = {frame}")        

        # Linha de trajetória do veículo a partir do frame
        df_traj["coords"] = df_traj.apply(lambda x:(x[x_column],x[y_column]),axis=1)
        df_traj["geometry"] = df_traj["coords"].apply(shapely.Point)
        # Cria o geodataframe
        df_traj = gpd.GeoDataFrame(df_traj,geometry="geometry",crs="EPSG:31984")
        
        # Geometria da linha
        traj_line = shapely.LineString(df_traj["coords"].tolist())
        # Ponto inicial
        start_point = df_traj["geometry"].values[0]
        # Instante inicial
        instant_time = df_traj[self.instant_column].values[0]
        
        # Verifica se a linha de trajetória cruza a linha de referencia
        if not traj_line.intersects(section):
            return -1,-1
        
        # Ponto de interseção
        intersection_point = traj_line.intersection(section)
        # Distância dos pontos de trajetoria ao ponto de interseção
        df_traj["distance_intersection"] = df_traj["geometry"].apply(lambda x:shapely.distance(x,intersection_point))
        df_traj = df_traj.sort_values("distance_intersection")
        
        # Instante de interseção
        instant_intersection = df_traj[self.instant_column].values[0]

        # Headway temporal do primeiro veículo em segundos
        hd1_time = instant_intersection - instant_time
        # Headway espacial do primeiro veículo em metros
        hd1_space = shapely.distance(start_point,intersection_point)

        return hd1_time,hd1_space
    
    def Hd1(self,frame,traffic_lane,section):
        """
        Calcula o headway temporal e espacial em relação à seção de referência
        """
        id_1st_vehicle = self.FirstVehicleQueue(frame,traffic_lane)

        # Se retornou np.nan
        if id_1st_vehicle==np.nan:
            return np.nan,np.nan
        hd1_time,hd1_space = self.TimeSpaceHeadway(id_1st_vehicle,frame,section)

        return hd1_time,hd1_space
    
    def Hd1FromEndMWA(self,frame,traffic_lane):
        """
        Calcula o hd1 a partir do frame fornecido e faixa indicada
        Considera a seção final do motobox
        """
        s = shapely.LineString([[self.motobox_end_section,self.video_heigth],[self.motobox_end_section,0]])
        h1d_time,h1d_space = self.Hd1(frame,traffic_lane,section=s)
        
        # Se retornou None,None, retorno None,None
        if (h1d_time==np.nan) and (h1d_space==np.nan):
            return np.nan,np.nan

        return h1d_time,h1d_space
    
    def HdFromEndMWA(self,id,frame):
        """
        Calcula o headway a partir do frame fornecido e faixa indicada
        Considera a seção final do motobox
        """
        # s = shapely.LineString([[self.motobox_end_section,self.video_heigth],[self.motobox_end_section,0]])
        s = shapely.LineString([[self.motobox_end_section,1000],[self.motobox_end_section,0]])
        hd_time,hd_space = self.TimeSpaceHeadway(id,frame,section=s)
        return hd_time,hd_space
    
    def Hd1Analysis(
            self,
            instant,
            side_offset_vehicle=0.15,
            dist_between_motorcycle_ahead:float=1,      # dbma
            dist_between_motorcycle_behind:float=2.5,   # dbmb
            max_long_dist_overlap=0.5,
            max_lat_dist_virtual_lane=0.7,
            max_distance_invading_section=2,
            frame_convert_mode="fps",
            fps=None,
            ):
        """
        Retorna um dataframe com variáveis necessárias para analisar o Hd1
        Retorna todas as faixas
        """
        if fps==None:
            fps = self.fps

        dbma = dist_between_motorcycle_ahead
        dbmb = dist_between_motorcycle_behind

        df_analysis = pd.DataFrame()

        # Ajuste do instante de tempo, tomando o mais próximo
        if frame_convert_mode not in ["snap","fps"]:
            raise ValueError(f"'frame_convert_mode' incorreto!")
        if frame_convert_mode=="snap":
            instant = self.df[self.instant_column].values[(self.df[self.instant_column]-instant).abs().argsort()[0]]
            # Frame de referência (preferencia)
            frame = self.df[self.df[self.instant_column]==instant][self.frame_column].values[0]
        if frame_convert_mode=="fps":
            frame = int(instant*fps)
        
        # Ids dos veículos nesse frame de cada fila
        df_analysis[self.traffic_lane_column] = self.traffic_lane_polygon["id"].values.astype(int)
        
        df_analysis["id1"] = df_analysis[self.traffic_lane_column].apply(lambda x:self.FirstVehicleQueue(frame,x))
        # Remove faixas que não contiverem veículos, para evitar erros futuros
        df_analysis = df_analysis.dropna(subset="id1")        

        # Se não houver veículos computáveis, retorna None
        if df_analysis.empty:
            return df_analysis
        
        # Puxa a posição de referência da frente do veículo, para alguns cálculos esécificos
        df_analysis[self.x_head_column] = df_analysis["id1"].apply(lambda x:self.df.query(f"{self.id_column} == {x} & {self.frame_column} == {frame}")[self.x_head_column].values[0])
        df_analysis["report"] = df_analysis.apply(lambda row:f"@{row['id1']} avançou a faixa de retenção por mais de {max_distance_invading_section} m" if row[self.x_head_column]>self.motobox_start_section+max_distance_invading_section else np.nan,axis=1)

        # Tipo de veículo
        df_analysis[self.vehicle_type_column] = df_analysis["id1"].apply(lambda x:self.FindVehicleType(x) if x!=None else "")
        # Veículo pesado ou não
        df_analysis["Qvp1j"] = df_analysis[self.vehicle_type_column].isin(self.vehicle_category_list['heavy']).astype(int)

        # Coluna para armasenar as motos armazenadas por ciclo
        df_analysis["idMcj"] = [[-1]]*len(df_analysis)

        # Quantidade de veículos a frente por categorias
        # IDs
        df_analysis["idQmfj"] = df_analysis["id1"].apply(lambda x:self.VehicleAhead(x,frame,side_offset_vehicle=side_offset_vehicle,max_longitudinal_distance_overlap=0.3)["id"].tolist())
        df_analysis["idQmfj"] = df_analysis.apply(lambda row:self.FilterVehicleOverlay(id_list=row["idQmfj"],frame=frame,max_centroid_overlap_dist=0.45),axis=1)
        df_analysis["idQmfXYj"] = df_analysis.apply(lambda row:[self.MotorcycleAheadFirstAnalysisDocAlessandro(i,frame,row[self.traffic_lane_column],max_long_dist_overlap=max_long_dist_overlap) for i in row["idQmfj"]],axis=1)
        # Contagens
        # Total
        df_analysis["Qmfj"] = df_analysis["idQmfXYj"].str.len()
        # Por categoria
        df_analysis["Qmf10j"] = [i.count('10') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf11j"] = [i.count('11') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf20j"] = [i.count('20') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf21j"] = [i.count('21') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf30j"] = [i.count('30') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf31j"] = [i.count('31') for i in df_analysis["idQmfXYj"]]
        # Restos
        df_analysis["Qmf00j"] = df_analysis["Qmfj"] - df_analysis[[
            'Qmf10j','Qmf11j',
            'Qmf20j','Qmf21j',
            'Qmf30j','Qmf31j',
            ]].sum(axis=1)
        
        # Redefine a quantidade agregada
        df_analysis["Qmfj"] = df_analysis[[
            'Qmf10j','Qmf11j',
            'Qmf20j','Qmf21j',
            'Qmf30j','Qmf31j',
            ]].sum(axis=1)
        
        # Unifica as colunas "idQmfj" e "idQmfXYj" em um dicionário
        df_analysis["idQmfj"] = df_analysis.apply(lambda row:dict(zip(row["idQmfj"],row["idQmfXYj"])),axis=1)
        df_analysis = df_analysis.drop(columns=["idQmfXYj"])

        # Atualiza as motos já contabilizadas por ciclo
        df_analysis["idMcj"] = df_analysis.apply(lambda row:row["idMcj"]+list(row["idQmfj"].keys()),axis=1)

        # Cálculo do headway (temporal e espacial)
        df_analysis[["hd1_time","hd1_space"]] = df_analysis.apply(lambda row:self.Hd1FromEndMWA(frame,row["traffic_lane"]),axis=1,result_type="expand")
        df_analysis["hd1_time"] = df_analysis["hd1_time"].round(2)
        df_analysis["hd1_space"] = df_analysis["hd1_space"].round(2)

        # Veículos nas laterais
        df_analysis["idQmcpj"] = df_analysis.apply(lambda row:self.SideVehicle(
            row["id1"],
            frame,
            max_lat_dist=max_lat_dist_virtual_lane,
            overlap_lon=-dbma,
            ignore_vehicle_types_list=self.vehicle_category_list["four_wheel"]).query(
                f"{self.x_tail_column} <= {row[self.x_head_column]}+{dbma} & {self.x_head_column} >= {row[self.x_head_column]}-{dbmb}"
            )[self.id_column].unique().tolist(),axis=1)
        
        # Remover ids já contabilizados em outras classes
        df_analysis[f"idQmcpj"] = df_analysis.apply(lambda row:[i for i in row[f"idQmcpj"] if i not in row["idMcj"]],axis=1)
        # Contabiliza
        df_analysis["Qmcpj"] = df_analysis["idQmcpj"].str.len()
        
        # Insere a informação de frame e instante de tempo
        df_analysis.insert(0,self.instant_column,instant)
        df_analysis.insert(0,self.instant_column+"_format",f"{int(instant//60)}:{int(instant%60)}")
        df_analysis.insert(0,self.frame_column,frame)

        last_frame = frame + int(df_analysis["hd1_time"].max()*self.fps)
        headway_sequence = self.GroupVechiclesCrossingSection(
            start_frame=frame,
            last_frame=last_frame,
            alignment_check=True
        )

        # Corrigir alinhamento
        for index,row in df_analysis.iterrows():
            alignment_set = headway_sequence[headway_sequence[self.id_column]==row["id1"]]
            if not alignment_set.empty:
                alignment_set = alignment_set["alignment"].values[0]
                df_analysis.loc[index,"alignment"] = alignment_set

        headway_sequence = headway_sequence.sort_values(by=["alignment",self.frame_column]).reset_index()
        
        for index,row in headway_sequence.iterrows():
            if index==0:
                headway_sequence.loc[index,"queue_position"] = 1
            elif row["alignment"]!=headway_sequence["alignment"].values[index-1]:
                headway_sequence.loc[index,"queue_position"] = 1
            else:
                headway_sequence.loc[index,"queue_position"] = headway_sequence["queue_position"].values[index-1] + 1
        
        hs = []
        for index,row in df_analysis.iterrows():
            value = headway_sequence[headway_sequence['alignment']==row["alignment"]]
            hd = [float(instant)]+value[self.instant_column].tolist()
            hd = [j-i for i,j in zip(hd[:-1],hd[1:])]
            value["hd"] = hd
            hs.append(value)

            if len(hd)>0:
                df_analysis.loc[index,"MaxHdj"] = [{int(value[value["hd"]==max(hd)][self.id_column].values[0]):round(max(hd),2)}]

        df_analysis["MaxHdj"] = df_analysis["MaxHdj"].apply(lambda value:value[0] if type(value)==list else np.nan)

        # Motobox modo Gambiarra
        df_analysis["MB"] = 0 if "SemMotobox" in self.processed_file else 1

        # Ajuste do tipo de variável
        df_analysis["id1"] = df_analysis["id1"].astype(int)

        return df_analysis
    
    def MotorcycleBetweenVehicleLF(
            self,
            id_vehicle_leader,
            id_vehicle_follower,
            frame,
            max_long_dist_overlap=0.3,
            side_offset_vehicle=0.15):
        """
        Retorna os ids das motocicletas entre os veículos líder e seguidor no frame indicado
        Ignora outros veículos não moto entre os veículos líder e seguidor
        Considera uma projeção linear a partir do veículo seguidor

        Retorna um dataframe com as motocicletas
        """

        # Dados de todos os veículos na seção
        df_analysis = self.df[self.df[self.frame_column]==frame]

        # Verifica se ambos os ids estão no frame
        if not (df_analysis[self.id_column] == id_vehicle_follower).any():
            raise ValueError(f"Id={id_vehicle_follower} não está no frame {frame}")
        if not (df_analysis[self.id_column] == id_vehicle_leader).any():
            raise ValueError(f"Id={id_vehicle_leader} não está no frame {frame}")
        
        # Verifica se os ids "id_vehicle_follower" e "id_vehicle_leader" são "não motos"
        vehicle_type_follower = self.FindVehicleType(id_vehicle_follower)
        if vehicle_type_follower in self.vehicle_category_list["two_wheel"]+self.vehicle_category_list["walk"]:
            raise ValueError(f"Id={id_vehicle_follower} é do tipo {vehicle_type_follower} (inválido)!")
        vehicle_type_leader = self.FindVehicleType(id_vehicle_leader)
        if vehicle_type_leader in self.vehicle_category_list["two_wheel"]+self.vehicle_category_list["walk"]:
            raise ValueError(f"Id={id_vehicle_leader} é do tipo {vehicle_type_leader} (inválido)!")
        
        # Definição dos limites longitudinais e laterais
        # Limites laterais e de do seguidor, definidos na função
        # Retorna todos os veículos sem limite longitudinal
        motorcycle_between = self.VehicleAhead(
                id_vehicle_follower,
                frame,
                side_offset_vehicle=side_offset_vehicle,
                max_longitudinal_distance_overlap=max_long_dist_overlap,
                ignore_vehicle_types_list=self.vehicle_category_list['four_wheel']+self.vehicle_category_list['walk'])
        
        # Restringe os veículos ao para-choque traseiro + sobreposição máxima
        # Próximo ao líder, o limite é o para-choque traseiro + sobreposição longitudinal
        lim_leader = df_analysis[df_analysis[self.id_column]==id_vehicle_leader][self.x_tail_column].values[0]
        lim_leader = lim_leader + max_long_dist_overlap

        # Restringe a quantidade de motocicletas, baseado na parte da frente dela
        motorcycle_between = motorcycle_between[motorcycle_between[self.x_head_column]<=lim_leader]

        return motorcycle_between

    def Hd4Analysis(
            self,
            instant,
            max_long_dist_overlap=0.5,
            side_offset_vehicle=0.15,
            dist_between_motorcycle_ahead:float=1,      # dbma
            dist_between_motorcycle_behind:float=2.5,   # dbmb
            lat_virtual_lane_overlap:float=0.4,
            max_distance_invading_section:float=2,
            max_lat_dist_virtual_lane:float=0.7,
            ):
        """
        Retorna um dataframe com variáveis necessárias para analisar
        o Hd4. Retorna todas as faixas
        """
        dbma = dist_between_motorcycle_ahead
        dbmb = dist_between_motorcycle_behind

        # Ajuste do instante de tempo, tomando o mais próximo
        instant = self.df[self.instant_column].values[(self.df[self.instant_column]-instant).abs().argsort()[0]]
        # Frame de referência (preferencia)
        frame = self.df[self.df[self.instant_column]==instant][self.frame_column].values[0]

        # Veiculos por fila
        # Mantém só first_group e agrupa por faixa
        df_analysis = self.QueueDetector(frame,distance_between_vehicles_lim=1e10).sort_values(by=[self.traffic_lane_column,"queue_position"])
        df_analysis[self.traffic_lane_column] = df_analysis[self.traffic_lane_column].astype(int)
        # Reduz a amostra para até 4 veículos
        df_analysis = df_analysis[df_analysis["queue_position"]<=4]
        # Indica se cada veículo é "pesado" ou não
        df_analysis["Qvpj"] = df_analysis[self.vehicle_type_column].isin(self.vehicle_category_list['heavy']).astype(int)
        df_analysis = df_analysis[df_analysis["first_group"]==1].groupby(self.traffic_lane_column).agg(
            {
                self.id_column:"unique",
                "Qvpj":"sum",
                self.x_head_column:"unique",
                self.x_tail_column:"unique"
            }
        ).reset_index(drop=False).rename(columns={self.id_column:"idj"})

        df_analysis["idj"] = df_analysis["idj"].apply(lambda x:x.tolist())

        # Cria um dataframe só com reports
        df_analysis["report"] = np.nan
        df_report = pd.DataFrame()

        # Contagem de veículos
        df_analysis["Qvfj"] = df_analysis["idj"].str.len()

        if not df_analysis[df_analysis["Qvfj"]>0].empty:
            # Id do primeiro veículo
            df_analysis["id1"] = df_analysis["idj"].str[0]

            # Tipo de veículo (pesado ou não) do primeiro veículo
            df_analysis[self.vehicle_type_column+"_1"] = df_analysis["id1"].apply(self.FindVehicleType)
            df_analysis["Qvp1j"] = df_analysis[self.vehicle_type_column+"_1"].isin(self.vehicle_category_list["heavy"]).astype(int)

            # Cálculo do headway (temporal e espacial) do primeiro veículo
            df_analysis[["hd1_time","hd1_space"]] = df_analysis.apply(lambda row:self.HdFromEndMWA(row["id1"],frame),axis=1,result_type="expand")
            df_analysis["hd1_time"] = df_analysis["hd1_time"].round(2)
            df_analysis["hd1_time"] = df_analysis["hd1_time"].round(2)
        
        # Report
        df_analysis["report"] = df_analysis["Qvfj"].apply(lambda row:f"@Fila insuficiente ({row})" if row<4 else np.nan)
        df_report = pd.concat([df_report,df_analysis[-df_analysis["report"].isna()]],ignore_index=True)
        # Remove faixas com menos de 4 veículos
        df_analysis = df_analysis[df_analysis["Qvfj"]>=4]

        # Se não houver faixas com veículos suficiente, com o dataframe vazio
        # Retorna a o dataframe vazio
        if df_analysis.empty:
            df_analysis = pd.concat([df_analysis,df_report],ignore_index=True)
            # Remove colunas inúteis
            df_analysis = df_analysis.drop(columns=[
                self.x_head_column,
                self.x_tail_column,])
            # Insere a informação de frame e instante de tempo para auxiliar na verificação
            df_analysis.insert(0,self.instant_column,instant)
            df_analysis.insert(0,self.instant_column+"_format",f"{int(instant//60)}:{int(instant%60)}")
            df_analysis.insert(0,self.frame_column,frame)
            return df_analysis,pd.DataFrame()

        # Último id
        df_analysis["id4"] = df_analysis["idj"].str[3]

        # Cálculo do headway (temporal e espacial) do último veículo
        df_analysis[["hd4_time","hd4_space"]] = df_analysis.apply(lambda row:self.HdFromEndMWA(row["id4"],frame),axis=1,result_type="expand")
        df_analysis["hd4_time"] = df_analysis["hd4_time"].round(2)
        df_analysis["hd4_time"] = df_analysis["hd4_time"].round(2)

        # Verifica se o 4 veículo passou na motofaixa
        # Report
        df_analysis["report"] = df_analysis.apply(lambda row:f"@{row['id4']} não passou na seção" if row["hd4_time"]<0 else np.nan,axis=1)
        df_report = pd.concat([df_report,df_analysis[-df_analysis["report"].isna()]],ignore_index=True)
        # Aplica o filtro
        df_analysis = df_analysis[df_analysis["hd4_time"]>=0]

        # Retorna a o dataframe vazio
        if df_analysis.empty:
            df_analysis = pd.concat([df_analysis,df_report],ignore_index=True)
            # Remove colunas inúteis
            df_analysis = df_analysis.drop(columns=[
                self.x_head_column,
                self.x_tail_column,])
            # Insere a informação de frame e instante de tempo para auxiliar na verificação
            df_analysis.insert(0,self.instant_column,instant)
            df_analysis.insert(0,self.instant_column+"_format",f"{int(instant//60)}:{int(instant%60)}")
            df_analysis.insert(0,self.frame_column,frame)
            return df_analysis,pd.DataFrame()

        # Verifica se o primeiro veículo está a frente da faixa de retenção até "max_distance_invading_section"
        # Report
        df_analysis["report"] = df_analysis.apply(lambda row:f"@{row['id1']} avançou a faixa de retenção por mais de {max_distance_invading_section} m" if row[self.x_head_column][0]>self.motobox_start_section+max_distance_invading_section else np.nan,axis=1)
        df_report = pd.concat([df_report,df_analysis[-df_analysis["report"].isna()]],ignore_index=True)
        # Aplica o filtro
        df_analysis = df_analysis[df_analysis[self.x_head_column].apply(lambda row: row[0]<=self.motobox_start_section+max_distance_invading_section)]

        # Se todos os veículos invadirem mais de "max_distance_invading_section"
        # Retorna a o dataframe vazio
        if df_analysis.empty:
            df_analysis = pd.concat([df_analysis,df_report],ignore_index=True)
            # Remove colunas inúteis
            df_analysis = df_analysis.drop(columns=[
                self.x_head_column,
                self.x_tail_column,])
            # Insere a informação de frame e instante de tempo para auxiliar na verificação
            df_analysis.insert(0,self.instant_column,instant)
            df_analysis.insert(0,self.instant_column+"_format",f"{int(instant//60)}:{int(instant%60)}")
            df_analysis.insert(0,self.frame_column,frame)
            return df_analysis,pd.DataFrame()

        # Coluna para armasenar as motos armazenadas por ciclo
        df_analysis["idMcj"] = [[-1]]*len(df_analysis)

        # Quantidade de veículos a frente por categorias
        # -----------------------------------------------------------------------------------------------------
        # IDs
        df_analysis["idQmfj"] = df_analysis["id1"].apply(lambda row:self.VehicleAhead(row,frame,side_offset_vehicle=side_offset_vehicle,max_longitudinal_distance_overlap=0.3)[self.id_column].tolist())
        df_analysis["idQmfj"] = df_analysis.apply(lambda row:self.FilterVehicleOverlay(id_list=row["idQmfj"],frame=frame,max_centroid_overlap_dist=0.45),axis=1)
        # Remover ids já contabilizados em outras classes
        df_analysis["idQmfj"] = df_analysis.apply(lambda row:[i for i in row["idQmfj"] if i not in row["idMcj"]],axis=1)

       
        # Classes
        df_analysis["idQmfXYj"] = df_analysis.apply(lambda row:[self.MotorcycleAheadFirstAnalysisDocAlessandro(i,frame,row[self.traffic_lane_column],max_long_dist_overlap=max_long_dist_overlap) for i in row["idQmfj"]],axis=1)
        # Contagens
        # Total
        df_analysis["Qmfj"] = df_analysis["idQmfXYj"].str.len()
        # Por categoria
        df_analysis["Qmf10j"] = [i.count('10') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf11j"] = [i.count('11') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf20j"] = [i.count('20') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf21j"] = [i.count('21') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf30j"] = [i.count('30') for i in df_analysis["idQmfXYj"]]
        df_analysis["Qmf31j"] = [i.count('31') for i in df_analysis["idQmfXYj"]]
        # Restos
        df_analysis["Qmf00j"] = df_analysis["Qmfj"] - df_analysis[[
            'Qmf10j','Qmf11j',
            'Qmf20j','Qmf21j',
            'Qmf30j','Qmf31j',
            ]].sum(axis=1)
        
        # Redefine a quantidade agregada
        df_analysis["Qmfj"] = df_analysis[[
            'Qmf10j','Qmf11j',
            'Qmf20j','Qmf21j',
            'Qmf30j','Qmf31j',
            ]].sum(axis=1)
        
        # Unifica as colunas "idQmfj" e "idQmfXYj" em um dicionário
        df_analysis["idQmfj"] = df_analysis.apply(lambda row:dict(zip(row["idQmfj"],row["idQmfXYj"])),axis=1)
        # -----------------------------------------------------------------------------------------------------

        # Atualiza as motos já contabilizadas por ciclo
        df_analysis["idMcj"] = df_analysis.apply(lambda row:row["idMcj"]+list(row["idQmfj"].keys()),axis=1)

        # Motos entre veículos
        # -----------------------------------------------------------------------------------------------------
        df_analysis["idQmevj"] = np.nan
        for pos in [1,2,3]:
            df_analysis[f"idQmev{pos}{pos+1}j"] = df_analysis.apply(lambda row:self.MotorcycleBetweenVehicleLF(
            id_vehicle_leader=row["idj"][pos-1],
            id_vehicle_follower=row["idj"][pos],
            frame=frame,
            max_long_dist_overlap=0.3,
            side_offset_vehicle=side_offset_vehicle)[self.id_column].unique().tolist(),axis=1)

            # Remover ids já contabilizados em outras classes
            df_analysis[f"idQmev{pos}{pos+1}j"] = df_analysis.apply(lambda row:[i for i in row[f"idQmev{pos}{pos+1}j"] if i not in row["idMcj"]],axis=1)

            # Quantidade por posição
            df_analysis[f"Qmev{pos}{pos+1}j"] = df_analysis[f"idQmev{pos}{pos+1}j"].str.len()
        
        # Contagem total
        df_analysis["Qmevj"] = df_analysis[[f"Qmev{i}{i+1}j" for i in [1,2,3]]].sum(axis=1)
        # Concatenação dos ids para verificação
        df_analysis["idQmevj"] = df_analysis.apply(lambda row:dict(zip(
            [str(i)+"-"+str(j) for i,j in zip(row["idj"][:3],row["idj"][1:4])],
            [row[f"idQmev{i}{i+1}j"] for i in [1,2,3]])),axis=1)
        # -----------------------------------------------------------------------------------------------------

        # Atualiza as motos já contabilizadas por ciclo
        df_analysis["idMcj"] = df_analysis.apply(lambda row:sum(list(row["idQmevj"].values()),row["idMcj"]),axis=1)

        # Motos no corredor (genérico)
        # -----------------------------------------------------------------------------------------------------
        
        df_analysis["idcvj"] = df_analysis[self.traffic_lane_column].apply(lambda row:[row,row+1])
        motorcycle_virtual_lane = self.MotorcycleInVirtualLane(frame,lat_virtual_lane_overlap=lat_virtual_lane_overlap)
        # Id por faixa
        df_analysis["idQvcj"] = df_analysis["idcvj"].apply(lambda row:motorcycle_virtual_lane[motorcycle_virtual_lane["virtual_traffic_lane"].isin(row)][self.id_column].tolist())
        # Remove os ids já contabilizados
        df_analysis["idQvcj"] = df_analysis.apply(lambda row:[i for i in row["idQvcj"] if i not in row["idMcj"]],axis=1)
        # Contagem
        df_analysis["Qvcj"] = df_analysis["idQvcj"].str.len()

        # Densidade de motocicletas = N / 2*comprimento do corredor
        # Multiplica por 2 pois são 2 corrdores possíveis por faixa
        df_analysis["CV_length"] = df_analysis[self.x_head_column].str[0] - df_analysis[self.x_tail_column].str[-1]
        df_analysis["Dcv"] = df_analysis["Qvcj"]/(df_analysis["CV_length"]*2)
        # -----------------------------------------------------------------------------------------------------

        # Veículos no corredor, próximo aos veículos
        # -----------------------------------------------------------------------------------------------------
        df_analysis["idQmcpj"] = np.nan
        for pos in [1,2,3,4]:
            df_analysis[f"idQmcp{pos}j"] = df_analysis.apply(lambda row:self.SideVehicle(
                row["idj"][pos-1],
                frame,
                overlap_lon=-dbma,
                max_lat_dist=max_lat_dist_virtual_lane,
                ignore_vehicle_types_list=self.vehicle_category_list["four_wheel"]).query(
                    f"{self.x_tail_column} <= {row[self.x_head_column][pos-1]}+{dbma} & {self.x_head_column} >= {row[self.x_head_column][pos-1]}-{dbmb}"
                )[self.id_column].unique().tolist(),axis=1)
            
            # Remove os ids já contabilizados
            df_analysis[f"idQmcp{pos}j"] = df_analysis.apply(lambda row:[i for i in row[f"idQmcp{pos}j"] if i not in row["idMcj"]],axis=1)
            # Quantidade por posição
            df_analysis[f"Qmcp{pos}j"] = df_analysis[f"idQmcp{pos}j"].str.len()
        
        # Contagem total
        df_analysis["Qmcpj"] = df_analysis[[f"Qmcp{i}j" for i in [1,2,3,4]]].sum(axis=1)
         # Concatenação dos ids para verificação
        df_analysis["idQmcpj"] = df_analysis.apply(lambda row:dict(zip([int(i) for i in row["idj"][:4]],[row["idQmcp1j"],row["idQmcp2j"],row["idQmcp3j"],row["idQmcp4j"]])),axis=1)
        # -----------------------------------------------------------------------------------------------------

        # Motobox modo Gambiarra
        df_analysis["MB"] = 0 if "SemMotobox" in self.processed_file else 1

        # Ajuste do tipo de variável
        df_analysis["id1"] = df_analysis["id1"].astype(int)
        df_analysis["id4"] = df_analysis["id4"].astype(int)

        # Headway entre---------------------------------------------------------------------------------
        last_frame = frame + int(df_analysis["hd4_time"].max()*self.fps)
        headway_sequence = self.GroupVechiclesCrossingSection(
            start_frame=frame,
            last_frame=last_frame,
            alignment_check=True
        )

        # Corrigir alinhamento
        for index,row in df_analysis.iterrows():
            alignment_set = headway_sequence[headway_sequence[self.id_column]==row["id1"]]

            if alignment_set.empty:
                alignment_set = headway_sequence[headway_sequence[self.id_column]==row["idj"][1]]
            alignment_set = alignment_set["alignment"].values[0]
            headway_sequence.loc[headway_sequence[self.id_column].isin(row["idj"]),"alignment"] = alignment_set
            df_analysis.loc[index,"alignment"] = alignment_set

        headway_sequence = headway_sequence.sort_values(by=["alignment",self.frame_column]).reset_index()
        
        for index,row in headway_sequence.iterrows():
            if index==0:
                headway_sequence.loc[index,"queue_position"] = 1
            elif row["alignment"]!=headway_sequence["alignment"].values[index-1]:
                headway_sequence.loc[index,"queue_position"] = 1
            else:
                headway_sequence.loc[index,"queue_position"] = headway_sequence["queue_position"].values[index-1] + 1
        
        hs = []
        for index,row in df_analysis.iterrows():
            value = headway_sequence[headway_sequence['alignment']==row["alignment"]]
            hd = [float(instant)]+value[self.instant_column].tolist()
            hd = [j-i for i,j in zip(hd[:-1],hd[1:])]
            value["hd"] = hd
            hs.append(value)

            df_analysis.loc[index,"MaxHdj"] = [{int(value[value["hd"]==max(hd)][self.id_column].values[0]):round(max(hd),2)}]
        
        df_analysis["MaxHdj"] = df_analysis["MaxHdj"].apply(lambda value:value[0])
        hs = pd.concat(hs,ignore_index=False)
        # ---------------------------------------------------------------------------------

        # Unir com os reports
        df_analysis = pd.concat([df_analysis,df_report],ignore_index=True).sort_values(self.traffic_lane_column)

        # Colunas temporárias removidas
        df_analysis = df_analysis.drop(columns=[
            self.x_head_column,
            self.x_tail_column,
            "idMcj",
            "idQmfXYj",
            "idQmcp1j",
            "idQmcp2j",
            "idQmcp3j",
            "idQmcp4j",
            "idQmev12j",
            "idQmev23j",
            "idQmev34j",
            "alignment"
            ])

        # Insere a informação de frame e instante de tempo para auxiliar na verificação
        df_analysis.insert(0,self.instant_column,instant)
        df_analysis.insert(0,self.instant_column+"_format",f"{int(instant//60)}:{int(instant%60)}")
        df_analysis.insert(0,self.frame_column,frame)

        return df_analysis,hs
    
    def MotorcycleAheadFirstAnalysisDocAlessandro(
            self,
            id,
            frame,
            traffic_lane,
            dist_between_motorcycle_vehicle_1:float=3.0,    # dbmv1
            dist_between_motorcycle_vehicle_2:float=4.5,    # dbmv2
            dist_between_motorcycle_motorcycle:float=1.5,   # dbmm
            max_long_dist_overlap:float=0.30,               # mldo
            side_offset_vehicle=0.3):          

        """
        Retorna a classe da motocicleta naquele frame
        10 - Moto com até "dbmv1" do carro de trás e sem outra moto até "dbmm" à frente
        11 - Moto com até "dbmv1" do carro de trás e com outra moto até "dbmm" à frente
        20 - Moto com até "dbmv2" do carro de trás e sem outra moto até "dbmm" à frente
        21 - Moto com até "dbmv2" do carro de trás e com outra moto até "dbmm" à frente
        30 - Moto com mais "dbmv2" do carro de trás e sem outra moto até "dbmm" à frente
        31 - Moto com mais "dbmv2" do carro de trás e com outra moto até "dbmm" à frente
        """

        # Utilizando variáveis com nomes menores
        dbmv1 = dist_between_motorcycle_vehicle_1
        dbmv2 = dist_between_motorcycle_vehicle_2
        dbmm = dist_between_motorcycle_motorcycle
        mldo = max_long_dist_overlap

        # Parte 1 das categorias
        crit1 = self.MotorcycleClassificationPt1(
            id,
            frame,
            traffic_lane,
            dist_between_motorcycle_vehicle_1=dbmv1,
            dist_between_motorcycle_vehicle_2=dbmv2,
            max_long_dist_overlap=mldo,
            side_offset_vehicle=side_offset_vehicle)
        
        # Parte 2 das categorias
        crit2 = self.MotorcycleClassificationPt2(
            id,
            frame,
            dist_between_motorcycle_motorcycle=dbmm,
            max_long_dist_overlap=mldo,
            side_offset_vehicle=0)
        
        return crit1+crit2
    
    def MotorcycleClassificationPt1(
            self,
            id,
            frame,
            traffic_lane,
            dist_between_motorcycle_vehicle_1:float=3.0,    # dbmv1
            dist_between_motorcycle_vehicle_2:float=4.5,    # dbmv2
            max_long_dist_overlap:float=0.30,               # mldo
            side_offset_vehicle=0.15,
            max_dist_invading_section=1.5):               
        
        # Utilizando variáveis com nomes menores
        dbmv1 = dist_between_motorcycle_vehicle_1
        dbmv2 = dist_between_motorcycle_vehicle_2
        mldo = max_long_dist_overlap

        # Distância do veículo de trás
        vehicle1 = self.VehicleBehind(
            id_vehicle=id,
            frame=frame,
            max_longitudinal_distance_overlap=mldo,
            side_offset_vehicle=side_offset_vehicle,
            ignore_vehicle_types_list=self.vehicle_category_list["two_wheel"])
        
        # Filtra a faixa
        vehicle1 = vehicle1[vehicle1[self.traffic_lane_column]==traffic_lane].sort_values("distance_between_vehicles")
        
        # Se não retornar veículos, retorna "0"
        if vehicle1.empty:
           return "0"
        
        # Ajusta para 1 veículo
        vehicle1 = vehicle1.iloc[:1]

        # Verifica se a moto veículo está além da seção do motobox mais folga
        # Posição da frente da moto no frame
        motorcycle_x_tail = self.df[(self.df[self.id_column]==id) & (self.df[self.frame_column]==frame)][self.x_tail_column].values[0]
        if motorcycle_x_tail>self.motobox_end_section:
            return "0"

        # Se até "dbmv1", retorna "1"
        if vehicle1["distance_between_vehicles"].values[0]<=dbmv1:
            return "1" 
        
        # Se até "dbmv1", retorna "2"
        if vehicle1["distance_between_vehicles"].values[0]<=dbmv2:
            return "2"

        # Se não for nenhum, retorna "3"
        return "3"
    
    def MotorcycleClassificationPt2(
            self,
            id,
            frame,
            dist_between_motorcycle_motorcycle:float=1.5,   # dbmm
            max_long_dist_overlap:float=0.30,               # mldo
            side_offset_vehicle=0):                      
        
        # Utilizando variáveis com nomes menores
        dbmm = dist_between_motorcycle_motorcycle
        mldo = max_long_dist_overlap

        # Distância do veículo de trás
        motorcycle2 = self.FirstVehicleAhead(
            id_vehicle=id,
            frame=frame,
            max_longitudinal_distance_overlap=mldo,
            side_offset_vehicle=side_offset_vehicle,
            ignore_vehicle_types_list=self.vehicle_category_list["four_wheel"])
        
        # Se não retornar veículos, retorna "0"
        if len(motorcycle2)<1:
           return "0"
        
        # Se até "dbmm", retorna "1"
        if motorcycle2["distance_between_vehicles"].values[0]<=dbmm:
            return "1" 

        # Se não for nenhum, retorna "0"
        return "0"
    
    def FindVehicleType(self,id):
        """
        Retorna o tipo de veículo com base no id
        Retorna a classe mais frequente, ao longo de todo o dataframe
        """
        vehicle_type = self.df.query(f"{self.id_column} == {id}").groupby(self.vehicle_type_column).agg({self.id_column:"count"}).sort_values(self.id_column,ascending=False).index[0]
        return vehicle_type

    # OBSOLETO
    def Tp1MotorcycleAnalysis(
        self,
        instant_reference:float,
        N:int=4,
        distance_between_motorcycle_and_vehicle_ahead_1:float=3.0,
        distance_between_motorcycle_and_vehicle_ahead_2:float=4.5,
        distance_between_motorcycle_and_motorcycle_ahead:float=1.5,
        distance_between_motorcycle_ahead_virtual_lane:float=1.0,
        distance_between_motorcycle_behind_virtual_lane:float=2.5,
        motobox_max_invasion_distance:float=1,
        max_transversel_distance_overlap_virtual_lane:float=0.30,
        max_longitudinal_distance_overlap:float=0.30,
        ghost_frames_generator_range=30,
        logs=True,
    ):
        '''
        Parâmetros padrão (unidade em metros)
        # Distancias a frente do primeiro veículo
        - Limites para o caso 1
        distance_between_motorcycle_and_vehicle_ahead_1 = 2.5
        - Limite para o caso 2
        distance_between_motorcycle_and_vehicle_ahead_2 = 4
        - Limites para o caso 2 e 3
        distance_between_motorcycle_and_motorcycle_ahead = 1

        # Distância longitudinal dos veiculos no corredor virtual
        - Limite a frente do veículo
        distance_between_motorcycle_ahead_virtual_lane = 0.5
        - Limite atrás da frente do veículo (campo de visão)
        distance_between_motorcycle_behind_virtual_lane = 2.5

        # Limite de invasão dos motobox pelos veículos
        motobox_max_invasion_distance = 1

        A função retorna um pd.DataFrame com os dados padronizados para o tp1
        '''
        # Futuramente pode ser alocada em outro canto isso
        # Geração de informações de frames possívelmente ausentes
        # Frame de referencia
        # Ajuste do instante de tempo
        instant_reference = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]
        frame_reference = self.df[self.df[self.instant_column]==instant_reference][self.frame_column].iloc[0]
        inf_frame = frame_reference-int(ghost_frames_generator_range/2)
        sup_frame = frame_reference+int(ghost_frames_generator_range/2)
        id_vehicle_list = self.df[self.df[self.instant_column].between(inf_frame,sup_frame)][self.id_column].tolist()

        # Geração de frames
        generated_ghost_frames = self.GhostFramesGenerator(id_vehicle_list,range_frame=(inf_frame,sup_frame))
        # Mesclar com df
        self.df = pd.concat([self.df,generated_ghost_frames],ignore_index=True).sort_values(by=[self.frame_column,self.id_column])

        # Inicio da função propriamente dita
        # Toma as filas formadas no ciclo
        df_queue = self.QueueDetector(instant_reference,distance_between_vehicles_lim=1e10)
        # Toma a primeira fila, mais proxima da faixa de retenção
        df_queue = df_queue[df_queue['first_group']==1]
        # Faixas
        traffic_lane_list = df_queue.sort_values(by=[self.traffic_lane_column,self.queue_position_column])[self.traffic_lane_column].unique().tolist()

        # Dados do tp1
        df_tp1 = pd.DataFrame()

        for traffic_lane in traffic_lane_list:
            valid_lane = True
            report = ''
            # Fila de veículos em uma determinada faixa
            df_analysed = df_queue[df_queue[self.traffic_lane_column]==traffic_lane]
            instant_min = str(int(instant_reference/60))+':'+(str(int(round(instant_reference % 60,0))) if len(str(int(round(instant_reference % 60,0))))>1 else '0'+str(int(round(instant_reference % 60,0))))

            # Exceções
            # Se houver a invasão de veículos de 4 rodas no motobox
            vehicle_motobox_invasion = df_analysed[df_analysed[self.x_head_column]>=self.motobox_start_section+motobox_max_invasion_distance]
            if len(vehicle_motobox_invasion)>0:
                report = report + '@' if len(report)>0 else ''
                report = report + f'O(s) veículo(s) {vehicle_motobox_invasion[self.id_column].tolist()} estão invadindo o motobox no instante {instant_min}'
                valid_lane = False

            # self.side_offset_vehicle_dict
            # Se nãou houver pelo menos N veículos na fila
            if len(df_analysed)<N:
                report = report + '@' if len(report)>0 else ''
                report = report + f'Tamanho de fila no instante {instant_min} na faixa {traffic_lane} é menor que {N}'
                valid_lane = False
            else:
                # --------------------------------------------------------------
                # Variável a ser explicada - headway acumulado do 4o veiculo
                idN = df_analysed[df_analysed[self.queue_position_column]==N].iloc[0][self.id_column]
                instant_crossing,instant_crossing_log = self.InstantCrossingSection(idN)
                H4j = instant_crossing - instant_reference

                # Se o headway for negativo, provavelmente o vídeo acabou antes do veículo cruzar a seção
                if H4j<0:
                    report = report + '@' if len(report)>0 else ''
                    report = report + f'O veículo {idN} não cruzou a faixa de interesse'
                    valid_lane = False

                # Verifica a distância entre cada veículo até a N posição
                # Calcula a fila real entre os veículos, coonsiderando motos entre veículos
                # Se alguma das distâncias calculadas for maior que 5m, não considera fila
                # Se alguma das distâncias for maior que 5m, verifica uma a uma, caso a
                # maior que 5m  não possua moto entre os veículos, exclui o ciclo
                pos_count = 1
                new_id = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.id_column]
                vehicle_type = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.vehicle_type_column]
                distance_to_vehicle_behind = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0]['distance_between_vehicles']
                # Veículo restritamente atrás
                vehicle_behind = self.FirstVehicleBehind(
                    new_id,
                    instant_reference,
                    side_offset_vehicle=self.side_offset_vehicle_dict[vehicle_type]*2
                )
                while pos_count<len(df_analysed):
                    if (distance_to_vehicle_behind<8) or (vehicle_behind.iloc[0]['distance_between_vehicles']<8):
                        pos_count = pos_count + 1
                        new_id = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.id_column]
                        vehicle_type = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0][self.vehicle_type_column]
                        distance_to_vehicle_behind = df_analysed[df_analysed[self.queue_position_column]==pos_count].iloc[0]['distance_between_vehicles']
                        # Veículo restritamente atrás
                        vehicle_behind = self.FirstVehicleBehind(
                            new_id,
                            instant_reference,
                            side_offset_vehicle=self.side_offset_vehicle_dict[vehicle_type]*2
                        )
                    else:
                        report = report + '@' if len(report)>0 else ''
                        report = report + f'Tamanho de fila no instante {instant_min} na faixa {traffic_lane} é menor que {N}, pois o veículo {new_id} está muito distante'
                        valid_lane = False
                        break

            if valid_lane:
                # --------------------------------------------------------------
                # Variável explicativa 1
                # Motcicletas à frente do primeiro veículo
                # Id do primeiro veiculo
                id1 = df_analysed[df_analysed[self.queue_position_column]==1].iloc[0][self.id_column]

                # Cálculos do H1
                instant_crossing,instant_crossing_log = self.InstantCrossingSection(id1)
                H1j = instant_crossing - instant_reference
                # Tipo de veículo referente ao primeiro
                Qvp1j = 1 if df_analysed[df_analysed[self.queue_position_column]==1].iloc[0][self.vehicle_type_column] in self.vehicle_category_list['heavy'] else 0

                # Todas as motocicletas à frente do priemiro veículo, incluindo lateralizadas
                all_motorcycles = self.VehicleAhead(id1,instant_reference,side_offset_vehicle=0.30).sort_values(by='distance_between_vehicles')
                motorcycles_ahead_projection = self.VehicleAhead(
                    id1,
                    instant_reference,
                    max_longitudinal_distance_overlap=max_longitudinal_distance_overlap
                    ).sort_values(by='distance_between_vehicles')

                # Todas as motocicletas
                idQmfj = all_motorcycles[self.id_column].tolist()
                motocycle_group_by_class = {
                    '10':[],
                    '11':[],
                    '20':[],
                    '21':[],
                    '30':[],
                    '31':[],
                }

                for id_motocycle in idQmfj:
                    group_motocycle_ahead = ''
                    # Motocicleta
                    vehicle = all_motorcycles[all_motorcycles[self.id_column]==id_motocycle]

                    if vehicle['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_vehicle_ahead_1:
                        group_motocycle_ahead = group_motocycle_ahead + '1'
                    elif vehicle['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_vehicle_ahead_2:
                        group_motocycle_ahead = group_motocycle_ahead + '2'
                    else:
                        group_motocycle_ahead = group_motocycle_ahead + '3'

                    motocycle_ahead = self.FirstVehicleAhead(id_motocycle,instant_reference,max_longitudinal_distance_overlap=max_longitudinal_distance_overlap)
                    # motocycle_ahead = all_motorcycles[]

                    if len(motocycle_ahead)>0:
                        if motocycle_ahead['distance_between_vehicles'].iloc[0]<=distance_between_motorcycle_and_motorcycle_ahead:
                            group_motocycle_ahead = group_motocycle_ahead + '1'
                        else:
                            group_motocycle_ahead = group_motocycle_ahead + '0'
                    else:
                        group_motocycle_ahead = group_motocycle_ahead + '0'

                    motocycle_group_by_class[group_motocycle_ahead].append(id_motocycle)

                idQmf10j = motocycle_group_by_class['10']
                idQmf11j = motocycle_group_by_class['11']
                idQmf20j = motocycle_group_by_class['20']
                idQmf21j = motocycle_group_by_class['21']
                idQmf30j = motocycle_group_by_class['30']
                idQmf31j = motocycle_group_by_class['31']

                idQmfj = idQmf10j+idQmf11j+idQmf20j+idQmf21j+idQmf30j+idQmf31j

                # --------------------------------------------------------------
                # Variável explicativa 2 - motos entre veiculos
                # Veículos da posição 2 a 4
                idQmevj = []

                for pos in range(2,N+1):
                    # # Posição correspondente a frente do carro da posição "pos"
                    # lim_distance_between_motorcycle_behind = df_analysed[df_analysed[self.queue_position_column]==pos][self.x_head_column].iloc[0]

                    # # Veíuclo na posição "pos - 1"
                    # vehicle = df_analysed[df_analysed[self.queue_position_column]==pos-1]

                    # # Motos atrás do veículo e com a traseira da moto à frente da posição da frente do veículo de trás
                    # motocycle_behind = self.VehicleBehind(vehicle[self.id_column].iloc[0],instant_reference,side_offset_vehicle=0.30,ignore_vehicle_types_list=self.vehicle_category_list['four_wheel'])

                    # motocycle_behind = motocycle_behind[motocycle_behind[self.x_tail_column]>=lim_distance_between_motorcycle_behind]

                    # idQmevjp = motocycle_behind[self.id_column].tolist()

                    # idQmevj.append(idQmevjp)

                    # Posição correspondente a atrás do carro da posição "pos-1"
                    lim_distance_between_motorcycle_ahead = df_analysed[df_analysed[self.queue_position_column]==pos-1][self.x_tail_column].iloc[0]
                    lim_distance_between_motorcycle_ahead = lim_distance_between_motorcycle_ahead + max_longitudinal_distance_overlap

                    # Veíuclo na posição "pos"
                    vehicle = df_analysed[df_analysed[self.queue_position_column]==pos]

                    # Motos a frnte do veículo e com a traseira da moto à frente da posição da frente do veículo de trás
                    # Mesmos critérios para a variável moto a frente do 1º veículo
                    motocycle_ahead = self.VehicleAhead(
                        vehicle[self.id_column].iloc[0],
                        instant_reference,
                        side_offset_vehicle=0.30,
                        max_longitudinal_distance_overlap=max_longitudinal_distance_overlap,
                        ignore_vehicle_types_list=self.vehicle_category_list['four_wheel'])

                    # Filtra as motos longitudinalmente
                    motocycle_ahead = motocycle_ahead[motocycle_ahead[self.x_head_column]<=lim_distance_between_motorcycle_ahead]

                    # Coleta o id
                    idQmevjp = motocycle_ahead[self.id_column].tolist()
                    idQmevj.append(idQmevjp)

                # --------------------------------------------------------------
                # Variável explicativa 3 - motos no corredor virtual
                # Limites longitudinais
                lim_first_vehicle = df_analysed[df_analysed[self.queue_position_column]==1].iloc[0][self.x_head_column] + distance_between_motorcycle_ahead_virtual_lane
                lim_N_vehicle = df_analysed[df_analysed[self.queue_position_column]==N].iloc[0][self.x_head_column] - distance_between_motorcycle_behind_virtual_lane
                # Todas as motos do correor virtual
                # motocycle_virtural_lane = self.PolygonalVeicleVirtualLane(
                #     instant_reference,
                #     max_transversel_distance_overlap=max_transversel_distance_overlap_virtual_lane)

                motocycle_virtural_lane = self.MotorcycleInVirtualLane(
                    instant_reference=instant_reference,
                    lat_virtual_lane_overlap=0.8,
                    include_motorcycle_box=True)

                # Filtra quais corredores são contabilizados
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.virtual_traffic_lane_column].isin([traffic_lane,traffic_lane+1])]

                # Filtra as motos nos limites longitudinais
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.x_tail_column]<=lim_first_vehicle]
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.x_head_column]>=lim_N_vehicle]

                # Ids das motos até o limite lateral das motos
                motorcycle_side = pd.concat([self.SideVehicle(id_vehicle,instant_reference,overlap_lat=max_transversel_distance_overlap_virtual_lane) for id_vehicle in df_analysed[df_analysed[self.queue_position_column].between(1,N)][self.id_column].tolist()])
                motorcycle_side = motorcycle_side[motorcycle_side['lateral_distance_between_vehicles'].abs()<=1][self.id_column].unique()
                motocycle_virtural_lane = motocycle_virtural_lane[motocycle_virtural_lane[self.id_column].isin(motorcycle_side)]

                # Ids das motos no corredor virtual
                # Densconsidera veículos previamente alocados entre veículos, e a frente, caso já exista
                idQmcvj = [i for i in motocycle_virtural_lane[self.id_column].tolist() if i not in JoinList(idQmevj)+idQmfj]


                # --------------------------------------------------------------
                # Variável explicativa 4 - Motos efetivamente impactando
                idQmcvpj = []
                for pos in range(1,N+1):
                    # Veiculo
                    vehicle = df_analysed[df_analysed[self.queue_position_column]==pos]

                    # Limites longitudinais
                    lim_ahead = vehicle.iloc[0][self.x_head_column] + distance_between_motorcycle_ahead_virtual_lane
                    lim_behind = vehicle.iloc[0][self.x_head_column] - distance_between_motorcycle_behind_virtual_lane

                    # Filtra as motos nos limites longitudinais
                    motocycle_virtural_lane_pos = motocycle_virtural_lane[motocycle_virtural_lane[self.x_tail_column]<=lim_ahead]
                    motocycle_virtural_lane_pos = motocycle_virtural_lane_pos[motocycle_virtural_lane_pos[self.x_head_column]>=lim_behind]

                    idQmcvpj.append(motocycle_virtural_lane_pos[self.id_column].tolist())


                idQmcvpj = [[j for j in i if j not in JoinList(idQmevj)+idQmfj] for i in idQmcvpj]
                # Primeiro veículo
                idQmcvp1j = idQmcvpj[0]

                # --------------------------------------------------------------
                # Variável explicativa 5 - Densidade de motocicletas no corredor virtual
                # Densidade do corredor virtual (2 corredores, cada um de cada lado)
                Dcvj = len(idQmcvj)/(2*(lim_first_vehicle-lim_N_vehicle))

                # --------------------------------------------------------------
                # Variável 6 - Quantidade de veículos pesados
                heavy_vehicle = df_analysed[(df_analysed[self.queue_position_column].between(1,N)) & (df_analysed[self.vehicle_type_column].isin(self.vehicle_category_list['heavy']))]
                idQvpj = heavy_vehicle[self.id_column].tolist()

                # --------------------------------------------------------------
                # Variável 7 - Motobox
                if self.motobox_start_section==self.motobox_end_section:
                    MB = 0
                else:
                    MB = 1

                summary_tp1 = {
                    'report':[report],
                    'frame':[frame_reference],
                    'instant':[instant_min],
                    'traffic_lane':[traffic_lane],
                    'idN':[idN],
                    'H4j':[round(H4j,2)],

                    'idQmevj':[idQmevj],
                    'Qmevj':[sum([len(i) for i in idQmevj])],

                    'idQmfj':[idQmfj],
                    'Qmfj':[len(idQmfj)],
                    'idQmf10j':[idQmf10j],
                    'Qmf10j':[len(idQmf10j)],
                    'idQmf20j':[idQmf20j],
                    'Qmf20j':[len(idQmf20j)],
                    'idQmf30j':[idQmf30j],
                    'Qmf30j':[len(idQmf30j)],
                    'idQmf11j':[idQmf11j],
                    'Qmf11j':[len(idQmf11j)],
                    'idQmf21j':[idQmf21j],
                    'Qmf21j':[len(idQmf21j)],
                    'idQmf31j':[idQmf31j],
                    'Qmf31j':[len(idQmf31j)],

                    'idQmcvj':[idQmcvj],
                    'Qmcvj':[len(idQmcvj)],

                    'idQmcvpj':[idQmcvpj],
                    'Qmcvpj':[sum([len(i) for i in idQmcvpj])],

                    'Dcvj':[round(Dcvj,2)],

                    'idQvpj':[idQvpj],
                    'Qvpj':[len(idQvpj)],

                    'MB':[MB],

                    "id1":[id1],
                    "H1j":[H1j],
                    "Qvp1j":[Qvp1j],
                    "idQmcvp1j":[[idQmcvp1j]],
                    "Qmcvp1j":[len(idQmcvp1j)]
                }

                df_tp1 = pd.concat([df_tp1,pd.DataFrame().from_dict(summary_tp1)],ignore_index=True)

            else:
                print(report) if logs else None
                summary_tp1 = {
                    'report':[report],
                    'frame':[frame_reference],
                    'instant':[instant_min],
                    'traffic_lane':[traffic_lane],
                    'idN':[[]],
                    'H4j':[0],

                    'idQmevj':[[]],
                    'Qmevj':[0],

                    'idQmfj':[[]],
                    'Qmfj':[0],
                    'idQmf10j':[[]],
                    'Qmf10j':[0],
                    'idQmf20j':[[]],
                    'Qmf20j':[0],
                    'idQmf30j':[[]],
                    'Qmf30j':[0],
                    'idQmf11j':[[]],
                    'Qmf11j':[0],
                    'idQmf21j':[[]],
                    'Qmf21j':[0],
                    'idQmf31j':[[]],
                    'Qmf31j':[0],

                    'idQmcvj':[[]],
                    'Qmcvj':[0],

                    'idQmcvpj':[0],
                    'Qmcvpj':[0],

                    'Dcvj':[0],

                    'idQvpj':[[]],
                    'Qvpj':[0],

                    'MB':[0],

                    "id1":[0],
                    "H1j":[0],
                    "Qvp1j":[0],
                    "idQmcvp1j":[[]],
                    "Qmcvp1j":[0]
                }

                df_tp1 = pd.concat([df_tp1,pd.DataFrame().from_dict(summary_tp1)],ignore_index=True)

        return df_tp1

    def LeaderFollowerVehiclePair(self,id_leader:int,id_follower:int,step=1):
        column_list = [
            self.instant_column,
            self.id_column,
            self.x_centroid_column,
            self.x_instant_speed_column,
            self.x_instant_acc_column,
        ]

        column_list = self.df.columns

        # Veiculos
        vehicle_leader = self.df[self.df[self.id_column]==id_leader]
        vehicle_follower = self.df[self.df[self.id_column]==id_follower]

        # Instantes válidos
        intersection_instant = sorted(list(set(vehicle_leader[self.instant_column]) & set(vehicle_follower[self.instant_column])))
        intersection_instant = [intersection_instant[i] for i in range(0,len(intersection_instant),step)]

        # Veiculos nos instantes válidos
        vehicle_leader = vehicle_leader[vehicle_leader[self.instant_column].isin(intersection_instant)]
        vehicle_follower = vehicle_follower[vehicle_follower[self.instant_column].isin(intersection_instant)]

        # Dados
        leader_follower_vehicle_pair = pd.DataFrame()
        leader_follower_vehicle_pair[self.instant_column] = intersection_instant

        # Unir com o líder
        leader_follower_vehicle_pair = pd.merge(leader_follower_vehicle_pair,
                                                vehicle_leader[column_list],
                                                how='left',
                                                on=self.instant_column,
                                                suffixes=('', '_leader')
                                               )

        # Unir com o seguidor
        leader_follower_vehicle_pair = pd.merge(leader_follower_vehicle_pair,
                                                vehicle_follower[column_list],
                                                how='left',
                                                on=self.instant_column,
                                                suffixes=('_leader', '_follower')
                                               )

        # Delta x gap
        leader_follower_vehicle_pair['delta_x_gap'] = leader_follower_vehicle_pair[self.x_tail_column+'_leader'] - leader_follower_vehicle_pair[self.x_head_column+'_follower']

        # Delta x headway
        leader_follower_vehicle_pair['delta_x_headway'] = leader_follower_vehicle_pair[self.x_head_column+'_leader'] - leader_follower_vehicle_pair[self.x_head_column+'_follower']

        # Delta x speed
        leader_follower_vehicle_pair['delta_speed_x'] = leader_follower_vehicle_pair[self.x_instant_speed_column+'_follower'] - leader_follower_vehicle_pair[self.x_instant_speed_column+'_leader']

        # TTC
        # Revisar esse parâmetro, acho que implementei errado
        leader_follower_vehicle_pair['TTC'] = leader_follower_vehicle_pair['delta_x_gap'] / leader_follower_vehicle_pair[self.x_instant_speed_column+'_follower']

        # Collision Degree
        leader_follower_vehicle_pair['collision_degree'] = ((leader_follower_vehicle_pair[self.y_centroid_column+'_follower']-leader_follower_vehicle_pair[self.y_centroid_column+'_leader'])/leader_follower_vehicle_pair['delta_x_gap']).apply(lambda x:np.rad2deg(np.arctan(x)))

        # Following headway 1
        # Headway (e não gap) projetado a cada time step
        leader_follower_vehicle_pair['following_headway_1'] = leader_follower_vehicle_pair['delta_x_headway'] / leader_follower_vehicle_pair[self.x_instant_speed_column+'_follower']

        # Following headway 2
        # Headway (e não gap) real a cada time step
        id_follower = leader_follower_vehicle_pair[self.id_column+'_follower'].iloc[0]
        fhd2_list = []
        for t in intersection_instant:
            new_section = leader_follower_vehicle_pair[leader_follower_vehicle_pair[self.instant_column]==t][self.x_head_column+'_leader'].iloc[0]
            fhd2, _ = self.InstantCrossingSection(id_follower,section_reference=new_section)
            fhd2_list.append(fhd2-t)

        leader_follower_vehicle_pair['following_headway_2'] = fhd2_list

        return leader_follower_vehicle_pair

    def CC0(self,instant_reference:float,side_offset_vehicle:float=None,ignore_vehicle_types_list:list=['Moto']):

        # Ajuste do instante de tempo
        instant_reference = self.df.iloc[(self.df[self.instant_column]-instant_reference).abs().argsort()[0]][self.instant_column]

        # Veículos no instante de referência
        vehicle = self.df[(-self.df[self.vehicle_type_column].isin(ignore_vehicle_types_list)) & (self.df[self.instant_column]==instant_reference)]

        # Dados CC0
        df_CC0 =pd.DataFrame()

        while len(vehicle)>0:
            vehicle_ahead = self.FirstVehicleAhead(vehicle[self.id_column].iloc[0],instant_reference=instant_reference,ignore_vehicle_types_list=ignore_vehicle_types_list)

            if len(vehicle_ahead)>0:
                motocycle_ahead = self.VehicleAhead(vehicle[self.id_column].iloc[0],instant_reference=instant_reference)
                motocycle_ahead = motocycle_ahead[motocycle_ahead[self.x_head_column]<=vehicle_ahead[self.x_tail_column].iloc[0]]

                df_CC0 = pd.concat([df_CC0,pd.DataFrame().from_dict({
                    self.instant_column:[instant_reference],
                    self.id_column+'_leader':[vehicle_ahead[self.id_column].iloc[0]],
                    self.vehicle_type_column+'_leader':[vehicle_ahead[self.vehicle_type_column].iloc[0]],
                    self.id_column+'_follower':[vehicle[self.id_column].iloc[0]],
                    self.vehicle_type_column+'_follower':[vehicle[self.vehicle_type_column].iloc[0]],
                    'motocyle_between':[len(motocycle_ahead)],
                    'CC0':[vehicle_ahead['distance_between_vehicles'].iloc[0]]
                })],ignore_index=True)

            vehicle = vehicle[1:]

        return df_CC0
    
    def FilterVehicleOverlay(self,id_list,frame,max_centroid_overlap_dist=0.25):
        """
        Analisa os veículos no frame indicado e verifica quais veiculos tem o
        centroide muito próximo ao de outros veículo
        Retorna o mesmo conjunto de id_list
        """
        # Deve ter pelo menos 2 veículos
        if len(id_list)<1:
            return id_list

        # Filtra os ids e frame
        df_analysis = self.df[(self.df[self.frame_column]==frame) & (self.df[self.id_column].isin(id_list))]
        
        # Cria o geodataframe com os dados
        df_analysis = gpd.GeoDataFrame(
            df_analysis,
            geometry=gpd.points_from_xy(
                df_analysis[self.x_centroid_column],
                df_analysis[self.y_centroid_column]),
            crs="EPSG:31984")
        
        # Compara com o própróprio dataframe, buscando o veículo mais próximo
        # Até a distância "max_centroid_overlap_dist", ignorando pontos sobrepostos
        df_analysis = df_analysis.sjoin_nearest(
            df_analysis[[self.id_column,"geometry"]],
            how="left",
            max_distance=max_centroid_overlap_dist,
            distance_col="distance_from_centroid",
            lsuffix="origin",
            rsuffix="overlap",
            exclusive=True)
        
        # Verifica se o "id_overlap" foi np.nan (são tem sobreposição)
        # Ou tem um valor, mantendo o menor id
        df_analysis = df_analysis[["id_origin","id_overlap","distance_from_centroid"]]
        df_analysis[self.id_column] = df_analysis.apply(lambda row:row["id_origin"] if row["id_overlap"]==np.nan else int(min(row["id_origin"],row["id_overlap"])),axis=1)

        # Retorna a lista final
        id_list = sorted(df_analysis[self.id_column].unique().tolist())
        return id_list
    
    def VechicleCrossingSection(
            self,
            vehicle_id,
            section,
            x_column=None,
            y_column=None,
            ):
        """
        Calcula em que momento o veículo cruza a seção
        Retorna a linha se cruzar e um dataframe vazio se não cruzar
        """

        # Ajusta as colunas padrão
        if x_column==None:
            x_column = self.x_head_column
        if y_column==None:
            y_column = self.y_centroid_column

        df_analysis = self.df[self.df[self.id_column]==vehicle_id].sort_values(self.frame_column)

        # Linha de trajetória do veículo a partir do frame
        df_analysis["coords"] = df_analysis.apply(lambda x:(x[x_column],x[y_column]),axis=1)
        df_analysis["geometry"] = df_analysis["coords"].apply(shapely.Point)
        # Cria o geodataframe
        df_analysis = gpd.GeoDataFrame(df_analysis,geometry="geometry",crs="EPSG:31984")
        
        # Geometria da linha
        df_analysis_line = shapely.LineString(df_analysis["coords"].tolist())
        
        # Verifica se a linha de trajetória cruza a linha de referencia
        if not df_analysis_line.intersects(section):
            return pd.DataFrame()
        
        # Ponto de interseção
        intersection_point = df_analysis_line.intersection(section)
        # Distância dos pontos de trajetoria ao ponto de interseção
        df_analysis["distance_intersection"] = df_analysis["geometry"].apply(lambda x:shapely.distance(x,intersection_point))
        df_analysis = df_analysis.sort_values("distance_intersection")

        row = pd.DataFrame(df_analysis.iloc[:1].drop(columns=["geometry","distance_intersection","coords"]))

        return row
    
    def VechicleCrossingFromEndMWA(
            self,
            vehicle_id,
            x_column=None,
            y_column=None,
            ):
        """
        Calcula em que momento o veículo cruza a seção
        Retorna a linha se cruzar e um dataframe vazio se não cruzar
        """

        # Ajusta as colunas padrão
        if x_column==None:
            x_column = self.x_head_column
        if y_column==None:
            y_column = self.y_centroid_column

        s = shapely.LineString([[self.motobox_end_section,self.video_heigth],[self.motobox_end_section,0]])
        row = self.VechicleCrossingSection(
            vehicle_id=vehicle_id,
            section=s,
            x_column=x_column,
            y_column=y_column
        )

        return row
    
    def GroupVechiclesCrossingSection(
        self,
        section=None,
        start_frame=None,
        last_frame=None,
        alignment_check=False,
        max_longitudinal_distance_overlap=0.3,
        **kwargs
        ):

        if section==None:
            section = shapely.LineString([[self.motobox_end_section,self.video_heigth],[self.motobox_end_section,0]])
        if start_frame==None:
            start_frame = 0
        if last_frame==None:
            last_frame = self.df[self.frame_column].max()
        
        x_column = None
        y_column = None
        if "x_column" in kwargs:
            x_column = kwargs["x_column"]
        if "y_column" in kwargs:
            y_column = kwargs["y_column"]
        if "test" in kwargs:
            print("sdsdstest")

        vehicle_id_list = self.df[self.df[self.frame_column].between(start_frame,last_frame)][self.id_column].unique().tolist()
        df = []
        for vehicle_id in vehicle_id_list:
            
            row = self.VechicleCrossingSection(
                vehicle_id=vehicle_id,
                section=section,
                x_column=x_column,
                y_column=y_column
            )
            df.append(row)
        
        df = pd.concat(df,ignore_index=True)
        # Filtra os limites para remover veículos que passaram na seção,
        # Mas fora dos frames indicados
        df = df[df[self.frame_column].between(start_frame,last_frame)].sort_values(self.frame_column)

        if alignment_check:
            count_alignment = 0
            df_dict = {
                self.id_column:[],
                "alignment":[],
                "queue_position":[]
            }
            for index,row in df.iterrows():
                if not row[self.id_column] in df_dict[self.id_column]:
                    count_pos = 1
                    count_alignment = count_alignment + 1

                    df_dict[self.id_column].append(row[self.id_column])
                    df_dict["alignment"].append(count_alignment)
                    df_dict["queue_position"].append(count_pos)

                    side_offset_vehicle = 0.1 if row[self.vehicle_type_column] in ["Moto","Bicicleta","Pedestre"] else -0.2
                    vehicle_behind = self.FirstVehicleBehind(
                        id_vehicle=row[self.id_column],
                        frame=row[self.frame_column],
                        side_offset_vehicle=side_offset_vehicle,
                        max_longitudinal_distance_overlap=max_longitudinal_distance_overlap
                    )
                    if not vehicle_behind.empty:
                        mask_vehicle_behind = df[self.id_column]==vehicle_behind[self.id_column].values[0]

                    while (not vehicle_behind.empty) and mask_vehicle_behind.any() and (not vehicle_behind[self.id_column].values[0] in df_dict[self.id_column]):
                        # Atualiza os dados do próximo veículo
                        count_pos = count_pos + 1
                        df_dict[self.id_column].append(vehicle_behind[self.id_column].values[0])
                        df_dict["alignment"].append(count_alignment)
                        df_dict["queue_position"].append(count_pos)
                        # print(row[self.id_column],vehicle_behind[self.id_column].values[0],mask_vehicle_behind.any(),count_alignment)

                        frame = df[mask_vehicle_behind][self.frame_column].values[0]
                        side_offset_vehicle = 0.1 if vehicle_behind[self.vehicle_type_column].values[0] in ["Moto","Bicicleta","Pedestre"] else -0.2

                        vehicle_behind = self.FirstVehicleBehind(
                            id_vehicle=vehicle_behind[self.id_column].values[0],
                            frame=frame,
                            side_offset_vehicle=side_offset_vehicle,
                            max_longitudinal_distance_overlap=max_longitudinal_distance_overlap
                            )
                        if not vehicle_behind.empty:
                            mask_vehicle_behind = df[self.id_column]==vehicle_behind[self.id_column].values[0]

            df = df.merge(pd.DataFrame.from_dict(df_dict),on=self.id_column,how="left").sort_values(by=["alignment","queue_position"])

        return df

# Fluxo de execução para trabalhar com múltiplos arquivos
# Copiar o padrão de alterar
class Run():
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
        run = Run()
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
    model.ImportFromJSON(file_path,post_processing=model.PostProcessing1)

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
    model.ImportFromJSON(file_path,post_processing=model.PostProcessing1)

    result = []
    hd = []
    for t in model.green_open_time:
        try:
            result_,hd_ = model.Hd4Analysis(t)
            if not result_.empty:
                result.append(result_)
                hd.append(hd_)
        except Exception as e:
            print(f"Erro {e} no {os.path.basename(file_path)} em {int(t//60)}:{int(t%60)}")

    result = pd.concat(result,ignore_index=True)
    hd = pd.concat(hd,ignore_index=True)

    result.insert(0,"file",os.path.basename(file_path))
    hd.insert(0,"file",os.path.basename(file_path))

    hd.to_csv(os.path.join("data/hd_check",os.path.basename(file_path).replace(".json",".csv")),index=False)

    return result

def RunDataProcessingFromParameterType1(file_path):
    start_timer = timeit.default_timer()
    print(f"Processando... {file_path}")

    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(file_path)

    if not os.path.exists(model.processed_file):

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
    else:
        print(model.processed_file, "já processado!")

def InsideCircle(x_center,y_center,x_object,y_object,radius):
    '''
    Determina se o ponto (x_object,y_object) está contido dentro do círculo cujo
    centro é x_center,y_center, com raio radius
    '''

    x_diff = x_center - x_object
    y_diff = y_center - y_object

    distance = ((x_diff**2) + (y_diff**2))**0.5

    return True if distance <= radius else False

def Intersection(list1,list2):
    if len(list1) == len(list2):
        min1 = min(list1)
        max1 = max(list1)
        min2 = min(list2)
        max2 = max(list2)

        range1 = np.arange(min1,max1,0.1)

        inter = 0
        for i in range1:
            if (i>=min2) and (i<=max2):
                inter = inter + 1

        return inter/len(range1)
    else:
        return 0

def AllNotPositive(list_values):
    for i in  list_values:
        if i <= 0:
            pass
        else:
            return False
    return True

def Null(list_values):
    list_values_pos = [-i if i < 0 else i for i in list_values]
    index_null = list_values_pos.index(min(list_values_pos))

    return list_values[index_null]

def PointToFunction(list_point,degree=2):
    x_points = np.array([x[0] for x in list_point])
    y_points = np.array([x[1] for x in list_point])
    curve = np.polyfit(x_points, y_points, degree)
    p = np.poly1d(curve)

    return p

def PolygonalToFunction(list_point):
    n = len(list_point)
    x_points = np.array([x[0] for x in list_point])
    y_points = np.array([x[1] for x in list_point])

    p = lambda point:np.piecewise(
        point+1e-20,
        [point < x_points[0]] + [(point >= x_points[i]) and (point < x_points[i+1]) for i in list(range(n))[:-1]] + [point >= x_points[-1]],
        [y_points[0]] + [PointToFunction([[x_points[i],y_points[i]],[x_points[i+1],y_points[i+1]]],degree=1)(point) for i in list(range(n))[:-1]] + [y_points[-1]]
        )

    return p

def JoinList(list_of_list):
    list_ = []
    for i in list_of_list:
        list_ = list_ + i

    return list_

def AnyValueInList(list1,list2):
    check = False
    if len(list1)>0:
        for i in list1:
            if i in list2:
                check = True
                break
            else:
                pass
    else:
        pass

    return check

def ConcatDicts(list_dicts,sort_values=False,ascending=True):
    """
    Concatena dicionários python, tratando vazios
    """
    result = {}
    for dct in list_dicts:
        if len(dct)>0:
            result.update(dct)
    
    if sort_values:
        result = dict(sorted(result.items(),key=lambda item:item[1],reverse=not ascending))
    return result