iterative_align(
    fq_in='/media/axel/d0a28364-6c64-4f8e-9efc-f332d9a0f1a9/SRR6675327_1.fastq.1000000.fastq',
    tmp_dir='/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/tmp2',
    ref='/media/axel/RSG5/data_celine/human_genome_Virus/human_set_viruses2',
    n_cpu=2,
    sam_out='/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/output/SRR6675327_1.fastq.1000000.sam',
    aligner="bowtie2",
    min_len=20,
    min_qual=30)

iterative_align(
    fq_in='/media/axel/d0a28364-6c64-4f8e-9efc-f332d9a0f1a9/SRR6675327_2.fastq.1000000.fastq',
    tmp_dir='/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/tmp2',
    ref='/media/axel/RSG5/data_celine/human_genome_Virus/human_set_viruses2',
    n_cpu=2,
    sam_out='/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/output/SRR6675327_2.fastq.1000000.sam',
    aligner="bowtie2",
    min_len=20,
    min_qual=30)
    
    
sam1="/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/output/SRR6675327_1.fastq.1000000.sam"
sam2="/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/output/SRR6675327_2.fastq.1000000.sam"
out_pairs = "/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/output/paired/test.pairs"
info_contigs="/media/axel/LaCie/other_human_datasets_2019/Arima_EBV/info_contigs.txt"
min_qual=30

threads=20

ps.sort("-@", str(threads), "-n", "-O", "SAM", "-o", sam1 + ".sorted", sam1)
st.move(sam1 + ".sorted", sam1)
ps.sort("-@", str(threads), "-n", "-O", "SAM", "-o", sam2 + ".sorted", sam2)
st.move(sam2 + ".sorted", sam2)

sam2pairs(sam1, sam2, out_pairs, min_qual)
