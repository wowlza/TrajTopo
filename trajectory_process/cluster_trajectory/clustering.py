# -*- coding: utf-8 -*-


import cv2
from trafficintelligence import moving, cvutils, storage
from sklearn.cluster import AffinityPropagation, KMeans
import sklearn.metrics as m
import shapely.geometry as SG
from shapely import affinity
import pickle
import os
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from shapely import LineString
import json
class Observation(object):
    '''
    Class to define a trajectory observation
    '''
    def __init__(self, num, traffic_intelligence_id, trajectory, intersection, trajectory_plot = None, f_star = None, approach = None, in_poly = None, length = None, num_points = 20):
        self.num = num
        self.ID = traffic_intelligence_id
        self.trajectory = trajectory
        # self.intersection = intersection
        self.trajectory_plot = trajectory_plot
        self.f_star = f_star
        self.num_points = num_points
        self.approach = approach
        self.in_poly = in_poly
        self.length = length
        
        self.get_approach()
        self.get_f_star()
        self.get_in_polygon()
        self.get_length()
                
    
    # def get_approach(self):
    #     '''Function to find the directional approach based on the approach polygons'''
    #     P = self.trajectory[0]
    #     for app, poly in self.intersection.approach_polys.items():
    #         if poly.contains(SG.Point(P.x,P.y)):
    #             self.approach = app
    #             continue
 
    
    # def get_in_polygon(self):
    #     '''return true if a trajectory starts or ends within a Shapely polygon'''
    #     P0 = self.trajectory[0]
    #     PL = self.trajectory[-1]
    #     if self.intersection.inner_poly.contains(SG.Point(P0.x,P0.y)) == False and self.intersection.inner_poly.contains(SG.Point(PL.x,PL.y)) == False:
    #         self.in_poly = False
    #     else:
    #         self.in_poly = True
            
    def get_length(self):
        '''get total distance travelled by the road user'''
        self.length = len(self.trajectory)

    
    # def get_f_star(self):
    #     '''locate position coordinates at given distance cut-off points'''
    #     self.trajectory.computeCumulativeDistances()
    #     Length=np.floor(self.trajectory.getCumulativeDistance(len(self.trajectory)-1))
    #     goal = np.array([X*Length/self.num_points for X in range(self.num_points)])
    #     distances = np.array([self.trajectory.getCumulativeDistance(pos) for pos in range(len(self.trajectory)-1)])
    #     indexes = abs(goal[:, None] - distances).argmin(axis=1)
    #     out = list(enumerate(indexes))
    #     reduced = []
    #     for pair in out:
    #         reduced.extend([self.trajectory[int(pair[1])].x, self.trajectory[int(pair[1])].y])
    #     self.f_star = reduced
    

cluster = {}

class Clusters(object):
    '''
    Class to store and cluster observations and output the results
    '''
    
    def __init__(self, trajectory_data, traj_min_length, num_points, trim = False, delete = False, num_SQL = 1000,  cluster_omit = 0, obs_list = [], af = None):
        # self.filedirectory = filedirectory
        # self.intersection = intersection
        self.trajectory_data = trajectory_data
        self.traj_min_length = traj_min_length
        self.num_points = num_points
        self.trim = trim
        self.delete = delete
        self.cluster_omit = cluster_omit
        self.obs_list =  obs_list
        self.af = af
        
        self.traj_length = 10
        
        self.load_observations_from_list()
        
    
    def load_observations_from_list(self):   
        '''
        Function to load observations from a list of trajectories, 
        where each trajectory is a list of 2D coordinates.
        '''
        
        for traj in self.trajectory_data:
            traj_linestring = LineString(traj)
            if traj_linestring.length > self.traj_min_length:
            # interpolate to the same length through shapely
                if len(traj) != self.traj_length: 
                    distances = np.linspace(0, traj_linestring.length, self.traj_length)
                    interpolated_traj = [traj_linestring.interpolate(distance).coords[0] for distance in distances]           
                    # traj_red = traj[:self.num_points] if self.trim else traj
                    # 检查裁剪后的轨迹是否仍满足长度要求
                    # if len(traj_red) > self.traj_min_length:
                    #     # 将有效轨迹添加到列表
                    self.obs_list.append(interpolated_traj)
       
                            
                            
    def find_optimal_preference(self, ss, clusters):
        '''function to find the preference value with the highest silhouette score and lowest number of clusters'''
        ss = np.asarray(ss)
        highest_ss = np.flatnonzero(ss == np.max(ss))                       #indices of highest silhouette scores
        _cluster = np.take(clusters, highest_ss)                            #number of clusters for highest silhouette score
        lowest_clusters = np.flatnonzero(clusters == np.min(_cluster))      #indices of lowest clusters 
        overlaps = [i for i in highest_ss if i in lowest_clusters]
        return overlaps[0]
    

    def cluster_trajectories(self, approach, segment_id, timestamp, plot = False, table = False):
        '''function to create A_star matrix and cluster using sklearn (Affinity Propagation)'''
        if len(self.obs_list) < 10:
            print("Not enough trajectories to cluster.")
            return
        # 使用轨迹坐标数据（A*矩阵）进行聚类
        # A_star = np.asarray(self.obs_list,dtype=object)
        A_star_1 = np.asarray(self.obs_list)
        A_star = A_star_1.reshape(A_star_1.shape[0], -1)
        X, ss, clusts = [], [], []
        # for x in range(-5000, -100, 100):
        for x in range(-50, -0, 5):
            af = AffinityPropagation(preference=x, random_state=1).fit(A_star)
            try:
                ss.append(m.silhouette_score(A_star, af.labels_))
                clusts.append(len(af.cluster_centers_indices_))
                X.append(x)
            except:
                continue
        if len(X) == 0:
            print('Clustering failed - possibly not enough valid trajectories.')
            return 
        
        # 找到最佳偏好值
        opt = self.find_optimal_preference(ss, clusts)
        print(f"Optimal silhouette score at index: {np.argmax(ss)}, preference: {X[opt]}")
        
        # 用最佳偏好值重新聚类
        self.af = AffinityPropagation(preference=X[opt], random_state=1).fit(A_star)
        silhouette_score = m.silhouette_score(A_star, self.af.labels_)
        
        if plot:
            self.plot_trajectories(self.obs_list, approach, segment_id, timestamp)
            
        if table:
            self.output_table(silhouette_score, segment_id, timestamp)
    
    def cluster_trajectories_KMeans(self, approach, segment_id, timestamp, plot=False, table=False):
        '''function to create A_star matrix and cluster using sklearn (KMeans)'''
        if len(self.obs_list) < 10:
            print("Not enough trajectories to cluster.")
            return

        # 使用轨迹坐标数据（A*矩阵）进行聚类
        A_star_1 = np.asarray(self.obs_list)
        A_star = A_star_1.reshape(A_star_1.shape[0], -1)  # 将轨迹数据拉伸为2D矩阵

        X, ss, clusts = [], [], []

        # 遍历不同的聚类数，替换 AffinityPropagation 为 KMeans
        for n_clusters in range(20, 40, 5):  # 从 2 到 10 个聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init='auto')
            try:
                labels = kmeans.fit_predict(A_star)  # 进行 KMeans 聚类
                silhouette_score = m.silhouette_score(A_star, labels)  # 计算轮廓系数
                ss.append(silhouette_score)  # 保存轮廓系数
                clusts.append(n_clusters)  # 保存聚类数量
                X.append(n_clusters)  # 保存聚类数
            except Exception as e:
                print(f"Clustering error with {n_clusters} clusters: {e}")
                continue

        if len(X) == 0:
            print('Clustering failed - possibly not enough valid trajectories.')
            return

        # 找到最佳聚类数（轮廓系数最大的聚类数）
        opt = np.argmax(ss)
        best_n_clusters = X[opt]
        print(f"Optimal silhouette score at index: {opt}, n_clusters: {best_n_clusters}")

        # 用最佳聚类数重新聚类
        self.kmeans = KMeans(n_clusters=best_n_clusters, random_state=1, n_init='auto').fit(A_star)
        silhouette_score = m.silhouette_score(A_star, self.kmeans.labels_)

        if plot:
            self.plot_trajectories_KMeans(self.obs_list, approach, segment_id, timestamp)

        if table:
            self.output_table(silhouette_score, segment_id, timestamp)      
        
       
     
    def output_table(self, silhouette_score, segment_id, timestamp):
        '''creates a json file with information about the clustered pathways'''
        Table = []
        for cluster in set(self.kmeans.labels_):
            if np.count_nonzero(self.kmeans.labels_ == cluster) < self.cluster_omit:
                continue
            # 假设聚类中心的特征展平后表示轨迹，恢复为 (10, 2) 的坐标矩阵
            cluster_center = self.kmeans.cluster_centers_[cluster].reshape(-1, 2)  # 将其转换为形状 (10, 2)
            Table.append(cluster_center.tolist())
        
        root = '/data/lza/cluster_traj_project/RoadUserPathways/cluster_results/val'
        folder = os.path.join(root, segment_id)
        os.makedirs(folder, exist_ok=True)
        file_name = os.path.join(f'{folder}/{timestamp}.json')
        with open(file_name, 'w') as json_file:
            json.dump(Table, json_file, indent=4)
       
                            
    
 
    def plot_trajectories(self, Full_traj, approach, segment_id, timestamp):
        '''function to plot all trajectories by cluster and plot the cluster exemplars with additional information'''
        import matplotlib.colors as colors
        
        # image_all = cv2.imread(self.filedirectory+'Geometry/plan.png')
        # image_dl = cv2.imread(self.filedirectory+'Geometry/plan.png')
        
        jet = plt.get_cmap('jet') 
        # cNorm  = colors.Normalize(vmin=0, vmax=len(self.af.cluster_centers_indices_))
        cNorm  = mcolors.Normalize(vmin=0, vmax=len(set(self.af.labels_)))  # Normalize colors for clusters
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        
        # colors = []
        # for idx in range(len(self.af.cluster_centers_indices_)):
        #     colorVal = scalarMap.to_rgba(idx)
        #     colorVal = [C*250 for C in colorVal] 
        #     colors.append(colorVal)
        
        # Generate a list of colors for each cluster
        # colors = [scalarMap.to_rgba(i) for i in range(len(set(self.af.labels_)))]
        
        # frequencies = [len(list(group)) for key, group in groupby(sorted(self.af.labels_))]
        # proportions = [round(clust*100/np.sum(frequencies),1) for clust in frequencies]
        
        # for cluster in set(self.af.labels_):
        #     if np.count_nonzero(self.af.labels_ == cluster) < self.cluster_omit:
        #         continue
        #     indicies = [i for i, D in enumerate(self.af.labels_) if D == cluster]
        #     for i in indicies:
        #         t = Full_traj[i]
        #         cvutils.cvPlot(image_all,t,colors[cluster%len(colors)],t.length())  
        #         if i in self.af.cluster_centers_indices_:
        #             cvutils.cvPlot(image_dl,t,colors[cluster%len(colors)],t.length(), thickness=2)   
        #             cv2.arrowedLine(image_dl, t[t.length()-2].asint().astuple(), t[t.length()-1].asint().astuple(), colors[cluster%len(colors)], thickness=2, tipLength=5)
        #             cv2.putText(image_dl,f'ID {cluster}/{frequencies[cluster]} obs/{proportions[cluster]}%',t[t.length()-np.random.randint(1,20)].asint().astuple(),cv2.FONT_HERSHEY_PLAIN,1,colors[cluster%len(colors)],thickness=2)
        # image_all = np.ones((200, 400, 3), np.uint8) * 255
        # image_dl = np.ones((200, 400, 3), np.uint8) * 255
        # Create two figures: one for all trajectories and one for important trajectories
        fig_all, ax_all = plt.subplots(figsize=(10, 10))
        fig_dl, ax_dl = plt.subplots(figsize=(10, 10))
        cmap = plt.get_cmap('jet')
        colors = [cmap(i / len(set(self.af.labels_))) for i in range(len(set(self.af.labels_)))]
        
        # Plot each cluster with a unique color
        for cluster in set(self.af.labels_):
            if np.count_nonzero(self.af.labels_ == cluster) < self.cluster_omit:
                continue
            indicies = [i for i, label in enumerate(self.af.labels_) if label == cluster]
            for i in indicies:
                traj = self.obs_list[i]
                traj = np.array(traj)
                traj = LineString(traj)
                
             # 保存绘图结果
            root = '/data/lza/cluster_traj_project/RoadUserPathways/vis_resluts/train'
            folder = os.path.join(root, segment_id)
            os.makedirs(folder, exist_ok=True)
            
            # # 创建第一个 figure ('all' 图像)
            # fig_all, ax_all = plt.subplots()
            # 在 'all' 图像上绘制所有轨迹（根据聚类结果给定颜色）
            ax_all.plot(*traj.xy, color=colors[cluster % len(colors)], linewidth=1)
            ax_all.text(3, 25, f'number trajectories: {len(indicies)}', fontsize=25, color='red')


            # 如果是聚类中心或重要轨迹，则在 'dl' 图像上绘制
            # # 创建第二个 figure ('dl' 图像)
            # fig_dl, ax_dl = plt.subplots()

            if i in self.af.cluster_centers_indices_:
                ax_dl.plot(*traj.xy, color=colors[cluster % len(colors)], linewidth=2)  # 使用更粗的线条绘制聚类中心轨迹
            # # Plot the trajectory on the 'all' image (show all trajectories by cluster)
            # cvutils.cvPlot(image_all, traj, colors[cluster % len(colors)], traj.shape[0])
            # # If it's a cluster center or an important trajectory, also plot on 'dl' image
            # if i in self.af.cluster_centers_indices_:  # Condition for important trajectories
            #     cvutils.cvPlot(image_dl, traj, colors[cluster % len(colors)], len(traj), thickness=2)
            #     # Add an arrow to indicate the direction of the trajectory
            #     # cv2.arrowedLine(image_dl, tuple(t[-2].astype(int)), tuple(t[-1].astype(int)), 
            #     #                 colors[cluster % len(colors)], thickness=2, tipLength=0.05)
            #     # # Add a label with the cluster ID, frequency, and proportion
            #     # cv2.putText(image_dl, f'ID {cluster}/{frequencies[cluster]} obs/{proportions[cluster]}%',
            #     #             tuple(t[len(t) - np.random.randint(1, 20)].astype(int)),
            #     #             cv2.FONT_HERSHEY_PLAIN, 1, colors[cluster % len(colors)], thickness=2)
            # # plt.plot(traj[:, 0], traj[:, 1], color=colors[cluster % len(colors)])      
        
        # plt.savefig(f'{folder}/{timestamp}_all.jpg', bbox_inches='tight')
        # plt.savefig(f'{folder}/{timestamp}_desire_lines.jpg', bbox_inches='tight')
        fig_all.savefig(f'{folder}/{timestamp}_all.jpg')
        fig_dl.savefig(f'{folder}/{timestamp}_desire_lines.jpg')
        plt.close('all') 
        # cv2.imwrite(self.filedirectory+f'output/{self.road_user_type}_{approach}_desire_lines.jpg', image_dl)
        # cv2.imwrite(self.filedirectory+f'output/{self.road_user_type}_{approach}_all.jpg', image_all)                
    def plot_trajectories_KMeans(self, Full_traj, approach, segment_id, timestamp):
        '''function to plot all trajectories by cluster and plot the cluster exemplars with additional information'''
        import matplotlib.colors as mcolors
        from shapely.geometry import LineString
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cmx
        
        # Create two figures: one for all trajectories and one for important trajectories
        fig_all, ax_all = plt.subplots(figsize=(10, 10))
        fig_dl, ax_dl = plt.subplots(figsize=(10, 10))
        
        cmap = plt.get_cmap('jet')  # Use jet colormap for clusters
        colors = [cmap(i / len(set(self.kmeans.labels_))) for i in range(len(set(self.kmeans.labels_)))]  # Adjust for KMeans labels

        # Plot each cluster with a unique color
        for cluster in set(self.kmeans.labels_):
            if np.count_nonzero(self.kmeans.labels_ == cluster) < self.cluster_omit:
                continue
            indicies = [i for i, label in enumerate(self.kmeans.labels_) if label == cluster]
            for i in indicies:
                traj = self.obs_list[i]
                traj = np.array(traj)
                traj = LineString(traj)  # Convert trajectory to LineString for easy plotting

                # 保存绘图结果
                root = '/data/lza/cluster_traj_project/RoadUserPathways/vis_resluts_KMeans/val'
                folder = os.path.join(root, segment_id)
                os.makedirs(folder, exist_ok=True)

                # 在 'all' 图像上绘制所有轨迹（根据聚类结果给定颜色）
                ax_all.plot(*traj.xy, color=colors[cluster % len(colors)], linewidth=1)
                plt.xlim(-50, 50)
                plt.ylim(-25, 25)
                # ax_all.text(3, 25, f'Number of trajectories: {len(indicies)}', fontsize=25, color='red')
                image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # 创建一个白色背景图像
                
                # 如果是聚类中心或重要轨迹，则在 'dl' 图像上绘制
                # 假设聚类中心的特征展平后表示轨迹，恢复为 (10, 2) 的坐标矩阵
                cluster_center = self.kmeans.cluster_centers_[cluster].reshape(-1, 2)  # 将其转换为形状 (10, 2)
                # 将轨迹数据 (cluster_center) 转换为整数坐标，因为 cv2 只接受整数坐标
                # cluster_center_int = np.int32(cluster_center)
                # # 使用虚线绘制聚类中心轨迹
                # for i in range(len(cluster_center_int) - 1):
                #     # 绘制虚线，间隔为 5 像素
                #     # coords convert
                #     cluster_center_int[i][0] = (cluster_center_int[i][0]+50)/0.5
                #     cluster_center_int[i][1] = (cluster_center_int[i][1]+25)/0.5
                    
                #     cluster_center_int[i+1][0] = (cluster_center_int[i][0]+50)/0.5
                #     cluster_center_int[i+1][1] = (cluster_center_int[i][1]+25)/0.5
                #     # if i % 2 == 0:  # 模拟虚线效果
                #     cv2.line(image, tuple(cluster_center_int[i]), tuple(cluster_center_int[i + 1]),
                #             color=(0, 0, 255), thickness=2)  # 使用红色线条 (BGR: (0, 0, 255)) 绘制
                # 绘制聚类中心的轨迹线
                ax_dl.plot(cluster_center[:, 0], cluster_center[:, 1], color=colors[cluster % len(colors)], linewidth=2)
                plt.xlim(-50, 50)
                plt.ylim(-25, 25)
                # 获取聚类中心轨迹的最后一个点
                # cluster_last_point = cluster_center[-1]  # 获取轨迹的最后一个 (x, y) 坐标
                # last_point = np.vstack(traj.xy)[:, -1]  # 将 traj.xy 转换为 2 x n 数组，并取最后一个点
                # if np.linalg.norm(cluster_last_point - last_point) < 1:  # Check if trajectory is near the cluster center
                #     ax_dl.plot(*traj.xy, color=colors[cluster % len(colors)], linewidth=2)  # 使用更粗的线条绘制聚类中心轨迹

        # 保存绘图
        fig_all.savefig(f'{folder}/{timestamp}_all_20-40-5_cluster_center.jpg')
        fig_dl.savefig(f'{folder}/{timestamp}_desire_lines_20-40-5_cluster_center.jpg')
        plt.close('all')
        # cv2.imwrite(f'{folder}/{timestamp}_cluster_center_trajectory.jpg', image)
    

class Intersection(object):
    '''
    Class to define the geometry of the infrastructure
    '''
    def __init__(self, filedirectory = None, inner_poly = None, outer_poly = None, center = None, arm_centers = None, mpp = None, approaches = None, approach_polys = None):
        self.filedirectory = filedirectory
        self.inner_poly = inner_poly
        self.outer_poly = outer_poly
        self.center = center
        self.arm_centers = arm_centers
        self.mpp = mpp                              #mpp is a meters per pixel value for the world image
        self.approaches = approaches
        self.approach_polys = approach_polys
        return
        
    def load_geometry(self, filedirectory):
        '''load existing geometrical information from given directory'''
        self.filedirectory = filedirectory
        self.inner_poly = SG.Polygon(np.load(filedirectory+'/innerBoundary.npy'))
        self.outer_poly = SG.Polygon(np.load(filedirectory+'/outerBoundary.npy'))
        self.center = np.load(filedirectory+'/intersectionCenter.npy')
        self.arm_centers = np.load(filedirectory+'/armCenters.npy')
        self.mpp = np.loadtxt(filedirectory+'/mpp.txt') 
        try:
            with open(filedirectory+'/approach_polys.pickle', 'rb') as handle:
                self.approach_polys = pickle.load(handle)
                self.plotPoly()           
        except:
            self.approach_polys = self.polys(directions = self.approaches, save = filedirectory+'/approach_polys.pickle') 
            self.plotPoly()
        return
    
    
    def point_input(self, image, info):
        '''function to get coordinate points from user clicking on image'''
        print(info[0])
        plt.figure(figsize=(5,5))
        plt.imshow(image)
        point = np.array(plt.ginput(info[1], timeout=3000))  
        plt.close('all')
        return point
        
    
    def define_geometry(self, filedirectory, approaches):
        '''write function to create polys needed for trimming and deleting'''
        self.filedirectory = filedirectory
        self.mpp = np.loadtxt(filedirectory+'/mpp.txt') 
        self.approaches = approaches
        image = plt.imread(filedirectory + '/plan.png')
        geometry = {'center': ['Click on the center of the intersection', 1, '/intersectionCenter.npy'],
                           'arm_centers': [f'Select a midpoint on approach arms {approaches} (starting with {approaches[0]})', len(approaches), '/armCenters.npy'],
                           'inner_poly':['Select four points of a polygon for deleting trajectories (inner_poly)', 4, '/innerBoundary.npy'],
                           'outer_poly':['Select four points of a polygon for trimming trajectories (outer_poly)', 4, '/outerBoundary.npy']}
        for key, info in geometry.items():
            points = self.point_input(image, info)*self.mpp
            np.save(filedirectory + info[2], points) 
        self.load_geometry(self.filedirectory)
        return


    def P_ave(self,p1,p2):
        '''return the midpoint of a line / average of two points'''
        return SG.Point((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)


    def plotPoly(self):
        '''plot polygons on world image and save image'''
        image = cv2.imread(self.filedirectory+'/plan.png')
        out_file = self.filedirectory+'/approach_polys.jpg'
        for app, poly in self.approach_polys.items():
            point_list = [moving.Point(point[0]/self.mpp,point[1]/self.mpp) for point in poly.exterior.coords]
            t = moving.Trajectory.fromPointList(point_list)
            cvutils.cvPlot(image,t,(255, 0, 0),thickness=2)   
            cv2.imwrite(out_file, image)


    def polys(self, directions = ['N','E','S','W'], save = False):
        '''create polygons from points in the middle of the approach arms and the center of the intersection.
        directions is input if it is not a four-approach intersection
        save is a file pathways'''
        C = self.center[0]
        arm_centers_extend = []
        for row in self.arm_centers:
            arm_centers_extend.append(row)   
        approaches = {}
        for c,arm in enumerate(arm_centers_extend):
            L1 = SG.LineString([C,arm])
            L2 = affinity.rotate(L1,45,origin=SG.Point(C))
            L3 = affinity.rotate(L1,-45,origin=SG.Point(C))
            approach_poly = SG.MultiPoint(list(L2.coords) + list(L3.coords)+ list(L1.coords))
            approaches[directions[c]] = approach_poly.convex_hull
        if save !=False:
            with open(save, 'wb') as handle:
                pickle.dump(approaches, handle)     
        return approaches
