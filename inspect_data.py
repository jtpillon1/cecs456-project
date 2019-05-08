def count(filename, numberofimages = 1462):

    global classes
    global number_in_class
    classes = []
    number_in_class = []

    path, dirs, files = next(os.walk("../" + filename + "/"))  

    for dir in dirs:
        path2, dirs2, files2 = next(os.walk("../" + filename + "/" + dir))  
        classes.append(dir)
        number_in_class.append(len(files2)-1)

    print(number_in_class)
    global number_training
    global number_tested
    number_training = sum(number_in_class)
    number_tested = numberofimages - number_training
    return classes  
    return number_in_class
    return number_training
    return number_tested

def graph(xlabel, ylabel, classes, number_in_class):   
    df = pd.DataFrame({xlabel:classes, ylabel:number_in_class})
    df = df.sort_values([ylabel], ascending=False)
    df.plot(kind='bar', x = xlabel, y = ylabel, legend=False, 
            color=['tab:blue','tab:orange', 'tab:green', 'tab:red', 
                   'tab:purple', 'tab:brown', 'tab:pink',
                   'tab:olive', 'tab:cyan'], width = 0.95)