#Make this into a pathway using flask between a html+CSS website and backend in python



import newsprocessor

def process_input(se):

    return se[::-1]

    
if __name__ == "__main__":
    search = input("Enter the Headline: ")
    print(process_input(search))
    