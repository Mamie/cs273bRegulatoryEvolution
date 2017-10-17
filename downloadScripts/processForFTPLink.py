def removeDuplicates(X):
    seen = set()
    uniq = []
    for x in X:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

if __name__=='__main__':
    with open('E-MTAB-2633.sdrf.txt', 'r') as infile:
        info = infile.readlines()
    info = list(info)[1:]
    info = [entry.split('\t') for entry in info]
    fastq = [entry[30].strip() for entry in info]
    fastq = removeDuplicates(fastq)
    processedRaw = [entry[34].strip() for entry in info]
    processed = []
    for entry in processedRaw:
        if bool(entry):
            processed.append(entry)
    processed = removeDuplicates(processed)        
    with open('fastq_ftp_links.txt', 'w') as outfile:
        for entry in fastq:
            outfile.write('%s\n' % entry)

    with open('processed_ftp_links.txt', 'w') as outfile:
        for entry in processed:
            outfile.write('%s\n' % entry)




