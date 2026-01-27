from dataset_scripts.analyze import MPIIGazeAnalyzer

def main():
    dataset_path = input("введите путь к MPIIGaze (enter for defaultpath): ").strip()
    
    if not dataset_path:
        print("default path ./MPIIGaze")
        dataset_path = "./MPIIGaze"
    
    analyzer = MPIIGazeAnalyzer(dataset_path)
    analyzer.scan()
    analyzer.analyze_annotations(1000)
    analyzer.analyze_quality(1000)
    analyzer.plot_do()
    analyzer.normalize_and_clean()
    analyzer.plot_posle()
    analyzer.report_autogeneration()

if __name__ == "__main__":
    main()