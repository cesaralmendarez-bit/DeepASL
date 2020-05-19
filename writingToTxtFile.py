class writeToFile:
    def toFile(output):
        f_report = open('outputFile.txt', 'a')
        letterToWrite = 'z'
        if output == 0:
            letterToWrite = 'A'
        elif output == 1:
            letterToWrite = 'L'
        elif output == 2:
            letterToWrite = 'B'
        elif output == 3:
            letterToWrite = 'C'
        elif output == 4:
            letterToWrite = 'H'

        f_report.write(letterToWrite)
        f_report.flush()
