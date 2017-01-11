library(lsa)

if ('wineFiles.rda' %in% list.files(path = 'data/')) {
    load('data/wineFiles.rda')
} else {
    words <- read.table("data/words.txt", header = FALSE, 
                        sep="\n", quote="", stringsAsFactors = FALSE)
    names <- read.table("data/names.txt", header = FALSE, 
                        sep="\n", quote="", stringsAsFactors = FALSE)
    weights <- read.csv("data/weights.csv", header = FALSE)
    weights <- data.matrix(weights)
    
    wineVectors <- read.csv("data/wineVectors.csv", header = FALSE)
    wineVectors <- data.matrix(wineVectors)
    
    wordVectors <- read.csv("data/wordVectors.csv", header = FALSE)
    wordVectors <- data.matrix(wordVectors)
    
    save(list = c("words","names","weights","wineVectors","wordVectors"), 
         file = "data/wineFiles.rda")
}

findClosestWines <- function(wineName, posAttr, negAttr, weights, wineVectors, 
                             names, wordVectors, words) {
    
    ## initialize the search vector
    searchVector <- matrix(0, 1, ncol(wineVectors))
    
    ## add in wine vector to search vector
    ## get weight from tfidf max
    if (class(wineName) == 'character') {
        nameIndex <- match(wineName, names[,,])
        weight <- weights[nameIndex]
        
        searchVector <- searchVector + wineVectors[nameIndex,]
    } else {
        weight <- 1
    }
    
    ## add in positive word vectors multiplied by weight
    if (class(posAttr) == 'character'){
        for (i in 1:length(posAttr)){
            wordIndex <- match(posAttr[i], words[,,])
            searchVector <- searchVector + weight * wordVectors[wordIndex,]
        }
    }
    
    ## subtract out negative word vectors multiplied by weight
    if (class(negAttr) == 'character'){
        for (i in 1:length(negAttr)){
            wordIndex <- match(negAttr[i], words[,,])
            searchVector <- searchVector - weight * wordVectors[wordIndex,]
        }
    }
    
    dists <- matrix(0, nrow(wineVectors), 1)
    
    for (i in 1:length(dists)){
        dists[i] <- cosine(as.vector(searchVector), wineVectors[i,])
    }
    
    return(dists)
}

shinyServer(
    function(input, output, session) {
        updateSelectizeInput(session, 'inWineName', 
                             choices = names[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions=3, 
                                          placeholder='Search for a wine')
                            )
        updateSelectizeInput(session, 'inPosWords', 
                             choices = words[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions=3, 
                                          placeholder='Add some attributes')
        )
        updateSelectizeInput(session, 'inNegWords', 
                             choices = words[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions=3, 
                                          placeholder='Add some attributes')
        )
        
        output$text1 <- renderText({
            paste(class(input$inWineName),
                  class(input$inPosWords),
                  class(input$inNegWords))
        })
    }
)