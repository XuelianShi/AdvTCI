import cv2
import numpy as np
import random
from tqdm import tqdm
import os
import math 

from detect_single_img import detect, detect_save

import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message='.*torch.cuda.amp.autocast.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

#PSO参数
omega = 0.9
c1 = 1.6
c2 = 1.4
r1 = 0.5
r2 = 0.6

#透明度
alpha = 0.5

#计数器
current_pso_image_count = 0
successful_attack_count = 0
detector_query_count = 0 

#用于存储每个符合PSO条件的图片的检测器查询次数的列表
detector_queries_per_pso_image = []

#population_size 是粒子群中的粒子数量
#X1, Y1, X2, Y2 定义了检测到的对象的边界框
def initialization(population_size, X1, Y1, X2, Y2):
    #每个粒子现在有9个维度：[r, g, b, x1, y1, x2, y2, x3, y3]
    population = np.zeros((population_size, 9))

    for i in range(population_size):
        particle_data = []
        
        #初始化 R, G, B 值 (0-255)
        r = random.uniform(0, 255)
        g = random.uniform(0, 255)
        b = random.uniform(0, 255)
        particle_data.extend([r, g, b])

        #在检测框内初始化三个随机点 (x, y)
        for _ in range(3): #生成3个点
            x = random.uniform(X1, X2)
            y = random.uniform(Y1, Y2)
            particle_data.extend([x, y])
            
        population[i] = np.array(particle_data)

    return population

def clip(particle_position, box):
    X1, Y1, X2, Y2 = box

    #将 R, G, B 值裁剪到 [0, 255]
    for i in range(3): #R, G, B 的索引为 0, 1, 2
        particle_position[i] = np.clip(particle_position[i], 0, 255)

    #将 x, y 坐标裁剪到检测框边界
    for i in range(3, 9, 2): # x 坐标的索引为 3, 5, 7
        particle_position[i] = np.clip(particle_position[i], X1, X2) #x坐标
        particle_position[i+1] = np.clip(particle_position[i+1], Y1, Y2) #y坐标

    return particle_position

#从粒子创建三角形点和颜色的函数
def create_triangle_and_color(particle_array):
    all_triangle_data = []

    for particle in particle_array:
        # particle: [r, g, b, x1, y1, x2, y2, x3, y3]
        r, g, b = int(particle[0]), int(particle[1]), int(particle[2])
        # OpenCV 的 BGR 格式
        color_bgr = (b, g, r)

        # 提取三个三角形顶点
        p1 = (particle[3], particle[4])
        p2 = (particle[5], particle[6])
        p3 = (particle[7], particle[8])

        
        # 转置为可识别的三个点
        triangle_points = np.array([p1, p2, p3], dtype=np.int32).reshape((-1, 1, 2))
        
        all_triangle_data.append((triangle_points, color_bgr))
        
    return all_triangle_data


def fitness_function(image, particle_position_list, detect_function, step_num, particle_idx, output_visualization_dir):
    global detector_query_count # 声明全局以修改计数器

    particle_position = particle_position_list[0]
    
    # 创建一个用于绘制三角形和透明度的临时图像
    temp_image = image.copy() 
    overlay = image.copy() 

    # 从粒子获取三角形点和颜色
    single_particle_data = create_triangle_and_color(np.array([particle_position]))

    current_triangle_points = None
    current_color_bgr = None
    current_area = 0

    for triangle_points, color_bgr in single_particle_data:
        current_triangle_points = triangle_points
        current_color_bgr = color_bgr

        if triangle_points is not None and len(triangle_points) > 0:
            # 在覆盖层上绘制填充的三角形
            cv2.fillPoly(overlay, [triangle_points], color_bgr)
        else:
            pass
    #加上透明度
    result_image_np = cv2.addWeighted(overlay, alpha, temp_image, 1 - alpha, 0)

    viz_filename = "process_image.jpg" 
    viz_filepath = os.path.join(output_visualization_dir, viz_filename)
    try:
        cv2.imwrite(viz_filepath, result_image_np)
    except Exception as e:
        print(f"Error saving visualization image to {viz_filepath}: {e}")

    # 在这里增加检测器查询计数，因为它用于适应度评估
    detector_query_count += 1
    # 调用 detect 函数，该函数返回 xmin, ymin, xmax, ymax, conf, shape
    # 将保存的图片路径传递给 detect_function
    xmin, ymin, xmax, ymax, conf, shape = detect_function(viz_filepath) 

    if shape == 0:
        fitness_score = 0.0
    else:
        fitness_score = conf

    # 输出每次适应度评估的进度
    print(f"Pic Num: {current_pso_image_count}, Count: {successful_attack_count}, "
          f"Query (current img): {detector_query_count}, PSO Step: {step_num}, Particle Idx: {particle_idx}, "
          f"Fitness: {fitness_score:.4f}, "
          )
    return fitness_score

def pso_optimization(image_np, box, detect_function, output_visualization_dir, population_size=50, max_steps=10, omega=0.9, c1=1.6, c2=1.4):
    X1, Y1, X2, Y2 = box

    # 每个粒子现在有9个维度
    population = initialization(population_size, X1, Y1, X2, Y2)

    velocities = np.zeros_like(population)
    P_best = np.copy(population)
    P_best_fitness = np.full(population_size, np.inf)

    G_best = np.copy(population[0])
    G_best_fitness = np.inf
    
    # 标记是否触发了提前停止
    early_stop_triggered = False
    
    fitness_values = np.zeros(population_size)

    # 存储实现 G_best_fitness 的结果图像，用于提前停止
    G_best_result_image = None
    

    for step_num in tqdm(range(max_steps), desc="PSO 优化"):
        
        for i in range(population_size):
        
            fitness_values[i] = fitness_function(image_np, [population[i]], detect_function, step_num, i, output_visualization_dir)
            

            if fitness_values[i] < P_best_fitness[i]:
                P_best[i] = population[i]
                P_best_fitness[i] = fitness_values[i]
            
            #在评估完该粒子的适应度后更新G_best
            if fitness_values[i] < G_best_fitness:
                G_best = population[i].copy() #创建副本
                G_best_fitness = fitness_values[i]
            
            # 如果当前粒子达到 0 适应度，立即提前停止
            if fitness_values[i] == 0.0:
                temp_image_for_early_stop = image_np.copy()
                overlay_for_early_stop = image_np.copy()
                        
                single_particle_data_early_stop = create_triangle_and_color(np.array([G_best]))
                for triangle_points_es, color_bgr_es in single_particle_data_early_stop:
                    if triangle_points_es is not None and len(triangle_points_es) > 0:
                        cv2.fillPoly(overlay_for_early_stop, [triangle_points_es], color_bgr_es)

                G_best_result_image = cv2.addWeighted(overlay_for_early_stop, alpha, temp_image_for_early_stop, 1 - alpha, 0)
                early_stop_triggered = True
                break # 跳出内部粒子循环

        if early_stop_triggered:
            break # 跳出外部步骤循环

        # 更新下一轮的速度和位置（仅在未提前停止的情况下）
        if not early_stop_triggered: #确保在提前停止后不更新粒子位置和速度
            for i in range(population_size):
                r1 = 0.5
                r2 = 0.6
                velocities[i] = (omega * velocities[i] +
                                 c1 * r1 * (P_best[i] - population[i]) +
                                 c2 * r2 * (G_best - population[i]))

                population[i] = population[i] + velocities[i]

                population[i] = clip(population[i], box) 
    
    #如果循环完成而没有提前停止，需要确保 G_best_result_image 已设置
    if G_best_result_image is None: #这会在没有提前停止时触发
        temp_image_final = image_np.copy()
        overlay_final = image_np.copy()
        single_particle_data_final = create_triangle_and_color(np.array([G_best]))
        for triangle_points_f, color_bgr_f in single_particle_data_final:
            if triangle_points_f is not None and len(triangle_points_f) > 0:
                cv2.fillPoly(overlay_final, [triangle_points_f], color_bgr_f)
        G_best_result_image = cv2.addWeighted(overlay_final, alpha, temp_image_final, 1 - alpha, 0)


    return G_best, G_best_result_image # 返回最佳粒子及其对应的图像

def process_image_with_pso(image_path, output_root_dir, detect_function, metrics):
    global current_pso_image_count, successful_attack_count, detector_query_count, detector_queries_per_pso_image

    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"错误: 无法从 {image_path} 加载图像。")
        return

    metrics['count_all'] += 1

    # 重置当前图片的 detector_query_count
    detector_query_count = 0

    # 初始检测也应计为一次查询
    detector_query_count += 1
    # 调用 detect 函数，该函数返回 xmin, ymin, xmax, ymax, conf, shape
    initial_xmin, initial_ymin, initial_xmax, initial_ymax, initial_conf, initial_shape = detect_function(image_path)
    # 仅当检测到一个对象时才继续
    if initial_shape != 1:
        print(f"跳过图片 {image_path}，初始检测到 {initial_shape} 个目标。")
        return

    current_pso_image_count += 1
    metrics['Query_eligible_images'] += 1 

    # 从返回值构建 detection_box
    detection_box = [initial_xmin, initial_ymin, initial_xmax, initial_ymax]
    

    # 为每张图片创建唯一的基于文件名的可视化目录
    image_base_name = "process_image.jpg"
    current_image_viz_dir = os.path.join(output_root_dir, "pso_viz", image_base_name)
    os.makedirs(current_image_viz_dir, exist_ok=True)
    
    
    optimized_particle, final_attack_image_np = pso_optimization(image_np, detection_box, detect_function,
                                                           current_image_viz_dir, 
                                                           population_size=50, max_steps=10, 
                                                           omega=0.9, c1=1.6, c2=1.4) 

    # 保存最终攻击图像
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_root_dir, f"result_{image_filename}")

    # 保存最终攻击图像到文件，然后将路径传递给 detect_function
    final_attack_image_save_path = os.path.join(output_root_dir, f"temp_final_attack_{image_filename}")
    cv2.imwrite(final_attack_image_save_path, final_attack_image_np)
    final_xmin, final_ymin, final_xmax, final_ymax, final_conf, final_shape = detect_function(final_attack_image_save_path)
    print(f"最终判断是否成功调用的检测器: xmin={final_xmin}, ymin={final_ymin}, xmax={final_xmax}, ymax={final_ymax}, conf={final_conf}, shape={final_shape}")
    detector_query_count += 1 # 最终检测检查也计为一次查询

    if final_shape == 0:
        print(f"成功！在处理后的图像 {output_path} 中未检测到任何目标。")
        metrics['ASR'] += 1
        successful_attack_count = metrics['ASR'] 
        cv2.imwrite(output_path, final_attack_image_np) 
        print(f"已将最终处理后的图像保存到 {output_path}")
    else:
        print(f"失败：在处理后的图像 {output_path} 中检测到 {final_shape} 个目标。")
        print(f"未能生成处理后的图像，可能是 PSO 优化失败。")

    
    # 存储该符合条件的图像的总查询次数
    detector_queries_per_pso_image.append(detector_query_count)


def main_processing_pipeline(input_image_dir, output_root_dir, detect_function):
    global current_pso_image_count, successful_attack_count, detector_query_count, detector_queries_per_pso_image

    current_pso_image_count = 0
    successful_attack_count = 0
    detector_query_count = 0 
    detector_queries_per_pso_image = []

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(os.path.join(output_root_dir, "pso_viz"), exist_ok=True) 
    metrics = {
        'count_all': 0, # 图片总数
        'Query_eligible_images': 0, 
        'ASR': 0 
    }

    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]


    for image_filename in image_files:
        image_path = os.path.join(input_image_dir, image_filename)
        process_image_with_pso(image_path, output_root_dir, detect_function, metrics)
        # Display cumulative metrics after each image's processing
        print(f"Total Images: {metrics['count_all']}, "
              f"pic_number: {metrics['Query_eligible_images']}, "
              f"ASR: {metrics['ASR']}")


    print("\n--- Final Results ---")
    print(f"Total images: {metrics['count_all']}")
    print(f"pic_Number: {metrics['Query_eligible_images']}")
    
    if metrics['Query_eligible_images'] > 0:
        total_queries_for_pso_images = sum(detector_queries_per_pso_image)
        average_queries_per_pso_image = total_queries_for_pso_images / metrics['Query_eligible_images']
        print(f"Total Queries: {total_queries_for_pso_images}")
        print(f"Average Queries: {average_queries_per_pso_image:.2f}")
        print(f"ASR Rate: {metrics['ASR'] / metrics['Query_eligible_images']:.4f}")
    else:
        print("error")
    
    print(f"ASR Count: {metrics['ASR']}")


if __name__ == "__main__":
    input_directory = '/root/autodl-tmp/.autodl/yolov5/TT00K-1/images' 
    output_directory = '/root/autodl-tmp/.autodl/output' 

    main_processing_pipeline(input_directory, output_directory, detect)