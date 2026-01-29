import os
from pathlib import Path
from collections import defaultdict

import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

class Analyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.persons = []
        self.data_stats = {
            'persons': {}, # участники
            'total_imgs': 0, # всего изображений
            'days_per_person': {}, 
            'imgs_per_day': defaultdict(list),
            'img_quality': {}, 
        }

# первичная проверка количества данных с которыми будем работать
    def scan(self):
        data_dir = self.dataset_path/'Data'/'Original'

        if not data_dir.exists():
            data_dir = self.dataset_path

        """
        person_dir - папка конкретного участника
        day_dir - папка дня участника
        """

        for person_dir in sorted(data_dir.iterdir()):
            if person_dir.is_dir() and person_dir.name.startswith('p'):
                person_id = person_dir.name
                self.persons.append(person_id)

                days = []
                total_imgs = 0

                for day_dir in sorted(person_dir.iterdir()):
                    if day_dir.is_dir and day_dir.name.startswith("day"):
                        days.append(day_dir.name)

                        imgs = list(day_dir.glob('*.jpg'))
                        num_imgs = len(imgs)
                        total_imgs += num_imgs
                        self.data_stats['imgs_per_day'][person_id].append(num_imgs)

                self.data_stats['persons'][person_id] = {
                    'total_imgs': total_imgs,
                    'num_days': len(days),
                    'days': days,
                }

                self.data_stats['total_imgs'] += total_imgs

        print(f"список участников {self.persons}") # список во всеми участниками
        print(f"общее количество файлов {self.data_stats['total_imgs']}") # количество файлов (фото)

    def analyze_annotations(self, sample_size: int):
        gaze_angles = []
        head_poses = []

        data_dir = self.dataset_path/'Data'/'Original'
        if not data_dir.exists():
            data_dir = self.dataset_path
        
        sampled = 0

        for participant_id in self.persons[:3]:  # первые три
            participant_dir = data_dir / participant_id
            
            for day_dir in participant_dir.iterdir():
                if day_dir.is_dir() and day_dir.name.startswith('day'):
                    annotation_file = day_dir / 'annotation.txt'
                    
                    if annotation_file.exists():
                        try:
                            annotations = np.loadtxt(annotation_file)
                            if len(annotations.shape) == 1:
                                annotations = annotations.reshape(1, -1)
                            
                            #gazetargets
                            if annotations.shape[1] >= 26:
                                gaze_angles.extend(annotations[:, 24:26].tolist())
                            
                            #headpose
                            if annotations.shape[1] >= 35:
                                head_poses.extend(annotations[:, 29:32].tolist())
                            
                            sampled += 1
                            if sampled >= sample_size:
                                break
                        except Exception as e:
                            print(f"error {e}")
                            continue
                
                if sampled >= sample_size:
                    break
            
            if sampled >= sample_size:
                break
        
        self.gaze_angles = np.array(gaze_angles) if gaze_angles else None
        self.head_poses = np.array(head_poses) if head_poses else None
        
        print(f"проанализировано аннотаций {sampled}")
        # print(f"{gaze_angles}")
        # print(f"{head_poses}")

    def analyze_quality(self, sample_size: int):
        brightness = []
        contrast = []
        resolutions = []

        data_dir = self.dataset_path/'Data'/'Original'
        if not data_dir.exists():
            data_dir = self.dataset_path

        sampled = 0
        for person_id in self.persons:
            person_dir = data_dir / person_id

            for day_dir in person_dir.iterdir():
                if day_dir.is_dir() and day_dir.name.startswith("day"):
                    imgs = list(day_dir.glob('*.jpg'))

                    # по 10 картинок иначе пиздец)
                    for img_path in imgs[:10]:
                        try:
                            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
                            if img is not None:
                                brightness.append(np.mean(img))
                                contrast.append(np.std(img))
                                resolutions.append(img.shape)
                                sampled += 1

                                if sampled >= sample_size:
                                    break

                        except Exception as e:
                            print(f"error {e}")
                            continue
                    
                    if sampled >= sample_size:
                        break

            if sampled >= sample_size:
                break        

        self.img_quality = {
            'brightness': brightness,
            'contrast': contrast,
            'resolutions': resolutions,
        }

        print(f"проанализировано фоток {sampled}")
        print(f"яркости изображений {brightness}\n")
        print(f"контраст изображений {contrast}\n")
        print(f"разрешения изображений {resolutions}\n") # 1280х720 for all

    def plot_do(self):
        # делим каждый отчет на 4 сабплота
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        persons_list = list(self.data_stats['persons'].keys())
        imgs_counts = [self.data_stats['persons'][p]['total_imgs'] for p in persons_list]
        days_counts = [self.data_stats['persons'][p]['num_days'] for p in persons_list]
        imgs_per_day_data = []
        for p in persons_list:
            imgs_per_day_data.extend(self.data_stats['imgs_per_day'][p])

        total_imgs = self.data_stats['total_imgs']
        total_persons = len(self.persons)
        avg_imgs_per_person = total_imgs / total_persons if total_persons > 0 else 0

        # отрисовка графиков ДО

        #1. количество изображений
        axes[0, 0].set_title("колво изобр до очистки")
        axes[0, 0].bar(range(len(persons_list)), imgs_counts)
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_xlabel("участник")
        axes[0, 0].set_ylabel("количество изображений")
        axes[0, 0].set_xticks(range(len(persons_list)))
        axes[0, 0].set_xticklabels(persons_list, rotation=45)

        #2. количество дней сьемки
        axes[1, 0].set_title("колво дней")
        axes[1, 0].bar(range(len(persons_list)), days_counts)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_xlabel("участник")
        axes[1, 0].set_ylabel("количество дней в который он фоткался")
        axes[1, 0].set_xticklabels(persons_list, rotation=45)
        axes[1, 0].set_xticks(range(len(persons_list)))

        #3. боксплот по дням сьемки
        stats_text = f"""
        ОБЩАЯ СТАТИСТИКА ДАТАСЕТА
        
        всего участников: {total_persons}
        всего изображений: {total_imgs:,}
        
        среднее на участника: {avg_imgs_per_person:.0f}
        минимум: {min(imgs_counts) if imgs_counts else 0:,}
        максимум: {max(imgs_counts) if imgs_counts else 0:,}
        """

        axes[0, 1].set_title("boxplot")
        axes[0, 1].boxplot([imgs_per_day_data], tick_labels=["все учатсники"])
        axes[0, 1].set_ylabel("фоток в день")
        axes[0, 1].grid(axis='x', alpha=0.3)


        #4. общий
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title("статистика ДО очистки")

        plt.tight_layout()
        plt.savefig('01_before_cleaning_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # if hasattr(self, 'img_quality') and self.img_quality['brightness']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # яркость
        axes[0].hist(self.img_quality['brightness'], bins=30)
        axes[0].set_xlabel("средняя яркость")
        axes[0].set_ylabel("частота")
        axes[0].set_title("рспределение яркости ДО")
        axes[0].grid(axis='y', alpha=0.3)
        
        # контраст
        axes[1].hist(self.img_quality['contrast'], bins=30)
        axes[1].set_xlabel("контраст")
        axes[1].set_ylabel("частота")
        axes[1].set_title("распределение контраста ДО")
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_before_cleaning_quality.png', dpi=300, bbox_inches='tight')
        plt.close()

    def normalize_and_clean(self):
        cleaned_stats = {
            'removed_low_brightness': 0,
            'removed_high_brightness': 0,
            'removed_low_contrast': 0,
            'total_removed': 0,
            'total_remaining': 0
        }
        
        if hasattr(self, 'img_quality') and self.img_quality['brightness']:
            brightness = np.array(self.img_quality['brightness'])
            contrast = np.array(self.img_quality['contrast'])
            
            # критерии
            brightness_low_threshold = np.percentile(brightness, 5)
            brightness_high_threshold = np.percentile(brightness, 95)
            contrast_threshold = np.percentile(contrast, 10)
            
            # подсчет удаленных изображений
            cleaned_stats['removed_low_brightness'] = np.sum(brightness < brightness_low_threshold)
            cleaned_stats['removed_high_brightness'] = np.sum(brightness > brightness_high_threshold)
            cleaned_stats['removed_low_contrast'] = np.sum(contrast < contrast_threshold)
            
            mask_remove = ((brightness < brightness_low_threshold) | 
                          (brightness > brightness_high_threshold) |
                          (contrast < contrast_threshold))
            
            cleaned_stats['total_removed'] = np.sum(mask_remove)
            cleaned_stats['total_remaining'] = len(brightness) - cleaned_stats['total_removed']
        
        self.cleaned_stats = cleaned_stats
        
        print(f"удалено изображений с низкой яркостью {cleaned_stats['removed_low_brightness']}")
        print(f"удалено изображений с высокой яркостью {cleaned_stats['removed_high_brightness']}")
        print(f"удалено изображений с низким контрастом {cleaned_stats['removed_low_contrast']}")
        print(f"всего удалено {cleaned_stats['total_removed']}")
        print(f"осталось {cleaned_stats['total_remaining']}")
        
        return cleaned_stats

    def plot_posle(self):
        fig, axes = plt.subplots(2, 2, figsize=(15,12))

        if hasattr(self, "img_quality") and self.img_quality['brightness']:
            brightness = np.array(self.img_quality['brightness'])
            contrast = np.array(self.img_quality['contrast'])
            
            brightness_low = np.percentile(brightness, 5)
            brightness_high = np.percentile(brightness, 95)
            contrast_threshold = np.percentile(contrast, 10)

            mask_keep = ~((brightness < brightness_low) |
                          (brightness > brightness_high) |
                          (contrast < contrast_threshold))
            
            #1. яркость до/после
            axes[0, 0].hist(brightness, bins=30, alpha=0.5, label="до очистки")
            axes[0, 0].hist(brightness[mask_keep], bins=30, alpha=0.7, label="после очистки")
            axes[0, 0].set_xlabel('средняя яркость')
            axes[0, 0].set_ylabel('частота')
            axes[0, 0].set_title('распределение яркости ДО и ПОСЛЕ')
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # 2. контраст до/после
            axes[0, 1].hist(contrast, bins=30, alpha=0.5, label="до очистки")
            axes[0, 1].hist(contrast[mask_keep], bins=30, alpha=0.7, label="после очистки")
            axes[0, 1].set_xlabel('контраст', fontsize=12)
            axes[0, 1].set_ylabel('частота', fontsize=12)
            axes[0, 1].set_title("распределение контраста ДО и ПОСЛЕ")
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            #3. статистика удаления
            categories = ['низкая яркость', 'высокая яркость', 'низкий контраст']
            values = [
                self.cleaned_stats['removed_low_brightness'],
                self.cleaned_stats['removed_high_brightness'],
                self.cleaned_stats['removed_low_contrast']
            ]
            
            axes[1, 0].bar(categories, values, color=['#ff9999', '#ffcc99', '#99ccff'])
            axes[1, 0].set_ylabel('количество удаленных изображений')
            axes[1, 0].set_title('причины удаления изображений')
            axes[1, 0].grid(axis='y', alpha=0.3)

            #4. общая стата после очистки
            stats_text = f"""
            РЕЗУЛЬТАТЫ ОЧИСТКИ
            
            изображений до очистки: {len(brightness):,}
            изображений удалено: {self.cleaned_stats['total_removed']:,}
            изображений осталось: {self.cleaned_stats['total_remaining']:,}
            
            процент удаленных: {(self.cleaned_stats['total_removed']/len(brightness)*100):.1f}%
            процент сохраненных: {(self.cleaned_stats['total_remaining']/len(brightness)*100):.1f}%
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            axes[1, 1].axis('off')
            axes[1, 1].set_title('сводка очистки')
        
        plt.tight_layout()
        plt.savefig('03_after_cleaning_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # автогенерация отчета по графикам в формате .md
    def report_autogeneration(self):
        report_path = 'dataset_report.md'

        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("# отчет по анализу датасета MPIIGaze\n\n")
            f.write(f"**дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            f.write("## 1. введение\n\n")
            f.write("MPIIGaze dataset that contains 213,659 images that we collected from 15 participants, ")
            f.write("during natural everyday laptop use over more than three months.")

            f.write("## 2. сбор и описание датасета\n\n")
            f.write("### 2.1 общая информация\n\n")
            f.write(f"- **количество участников:** {len(self.persons)}\n")
            f.write(f"- **общее количество фото:** {self.data_stats['total_imgs']:,}\n")
            
            if self.persons:
                avg_images = self.data_stats['total_imgs'] / len(self.persons)
                f.write(f"- **среднее количество изображений на участника:** {avg_images:.0f}\n\n")
            
            f.write("### 2.2 структура датасета\n\n")
            f.write("датасет организован по следующей схеме\n\n")
            f.write("```\n")
            f.write("MPIIGaze/\n")
            f.write("├── Data/\n")
            f.write("│   ├── Original/\n")
            f.write("│   │   ├── p00/\n")
            f.write("│   │   │   ├── day01/\n")
            f.write("│   │   │   │   ├── 0001.jpg\n")
            f.write("│   │   │   │   ├── annotation.txt\n")
            f.write("│   │   │   ├── day02/\n")
            f.write("│   │   ├── p01/\n")
            f.write("│   │   ...\n")
            f.write("│   └── Normalized/\n")
            f.write("```\n\n")
            
            f.write("## 3. анализ датасета ДО очистки\n\n")
            f.write("### 3.1 распределение данных\n\n")
            f.write("![распределение изображений](01_before_cleaning_overview.png)\n\n")
            
            f.write("### 3.2 статистика по участникам\n\n")
            f.write("| участник | изображений | дней съемки | среднее/день |\n")
            f.write("|----------|-------------|-------------|---------------|\n")
            
            for p_id in sorted(self.data_stats['persons'].keys()):
                p_data = self.data_stats['persons'][p_id]
                avg_per_day = p_data['total_imgs'] / p_data['num_days'] if p_data['num_days'] > 0 else 0
                f.write(f"| {p_id} | {p_data['total_imgs']:,} | {p_data['num_days']} | {avg_per_day:.1f} |\n")
            
            f.write("\n")
            
            if hasattr(self, 'img_quality') and self.img_quality['brightness']:
                f.write("### 3.3 качество изображений\n\n")
                f.write("![качество изображений](02_before_cleaning_quality.png)\n\n")
                
                brightness = np.array(self.img_quality['brightness'])
                contrast = np.array(self.img_quality['contrast'])
                
                f.write("**статистика яркости:**\n")
                f.write(f"- среднее: {np.mean(brightness):.2f}\n")
                f.write(f"- медиана: {np.median(brightness):.2f}\n")
                f.write(f"- стд. откл.: {np.std(brightness):.2f}\n")
                f.write(f"- мин/макс: {np.min(brightness):.2f} / {np.max(brightness):.2f}\n\n")
                
                f.write("**статистика контраста:**\n")
                f.write(f"- среднее: {np.mean(contrast):.2f}\n")
                f.write(f"- седиана: {np.median(contrast):.2f}\n")
                f.write(f"- стд. откл.: {np.std(contrast):.2f}\n")
                f.write(f"- мин/сакс: {np.min(contrast):.2f} / {np.max(contrast):.2f}\n\n")
            
            f.write("## 4. очистка и нормализация датасета\n\n")
            f.write("### 4.1 критерии очистки\n\n")
            f.write("для удаления изображений низкого качества я применил фильтрацию по критериям:\n\n")
            f.write("1. **экстремальная яркость:** с яркостью < 5% и > 95%\n")
            f.write("2. **низкий контраст:** с контрастом < 10%\n")
            f.write("3. **дубликаты:**\n\n")
            
            if hasattr(self, 'cleaned_stats'):
                f.write("### 4.2 результаты очистки\n\n")
                f.write(f"- **удалено из-за низкой яркости:** {self.cleaned_stats['removed_low_brightness']}\n")
                f.write(f"- **удалено из-за высокой яркости:** {self.cleaned_stats['removed_high_brightness']}\n")
                f.write(f"- **удалено из-за низкого контраста:** {self.cleaned_stats['removed_low_contrast']}\n")
                f.write(f"- **всего удалено:** {self.cleaned_stats['total_removed']}\n")
                f.write(f"- **осталось изображений:** {self.cleaned_stats['total_remaining']}\n\n")
            
            f.write("## 5. анализ датасета ПОСЛЕ очистки\n\n")
            f.write("### 5.1 сравнение до/после\n\n")
            f.write("![сравнение до и после очистки](03_after_cleaning_comparison.png)\n\n")
            
            f.write("## 6. технические детали\n\n")
            f.write("### 6.1 used libs\n\n")
            f.write("- numpy\n")
            f.write("- pandas\n")
            f.write("- matplotlib\n")
            f.write("- seaborn\n")
            f.write("- opencv2\n\n")
            
            f.write("---\n\n")
            f.write("*отчет сгенерирован автоматически*")

        return report_path
    

    def activate(self):
        self.scan()
        self.scan()
        self.analyze_annotations(500)
        self.analyze_quality(10000)
        self.plot_do()
        self.normalize_and_clean()
        self.plot_posle()
        self.report_autogeneration()