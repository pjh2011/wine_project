#server.R

library(lsa)

#load data
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

# function to find the closest wines by cosine distance
# can search by either a wine, a composition of words, or a
# combination of the two
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
    
    wines <- names[rev(tail(order(dists), 11)),]
    
    if (class(wineName) == 'character'){
        if (wineName == wines[1]) {
            return(wines[2:11])
        }
    }
    
    return(wines[1:10])
}

shinyServer(
    function(input, output, session) {
        updateSelectizeInput(session, 'inWineName', 
                             choices = names[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions = 3,
                                          maxItems = 1,
                                          placeholder = 'Search for a wine')
                            )
        updateSelectizeInput(session, 'inPosWords', 
                             choices = words[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions = 3, 
                                          placeholder = 'Add some attributes')
        )
        updateSelectizeInput(session, 'inNegWords', 
                             choices = words[,'V1'], 
                             server = TRUE, 
                             options=list(maxOptions = 3, 
                                          placeholder = 'Add some attributes')
        )
        
        output$similarWines <- renderUI({
            wineName <- input$inWineName
            posWords <- input$inPosWords
            negWords <- input$inNegWords
            
            entryExists <- (class(wineName) == 'character') | 
                (class(posWords) == 'character') | 
                (class(negWords) == 'character')
            
            if (entryExists) {
                wines <- findClosestWines(wineName, posWords, negWords, 
                                          weights, wineVectors, names, 
                                          wordVectors, words)
                HTML(paste('<h3>Top Wine Matches:</h3><ol>', 
                           paste('<li>',wines,'</li>', collapse = ''), 
                           '</ol>'))   
            } else {
                HTML('')
            }
            
            # findClosestWines(NULL, c('red'), NULL, weights, wineVectors, names, wordVectors, words)
            #paste(class(input$inWineName),
            #      class(input$inPosWords),
            #      class(input$inNegWords))
            # http://stackoverflow.com/questions/23233497/outputting-multiple-lines-of-text-with-rendertext-in-r-shiny
            # http://stackoverflow.com/questions/22923784/how-to-add-bullet-points-in-r-shinys-rendertext
        })
        
        output$descr1 <- renderText({
            paste('Welcome to the Wine2Vec search tool! This app will allow',
                    'you to search a database of over 30 thousand bottles of',
                    'wine. Check out the Search Wine Database page to give it',
                    'a try.')
        })
        
        output$descr2 <- renderText({
            paste('In order to search you can select a wine, then optionally',
                  'add and subtract characteristics that you enjoy or',
                  'dislike. Then Wine2Vec will find the most similar bottles.',
                  'Or you can just list some characteristics',
                  'you enjoy and/or dislike and find wines that match your',
                  'description.')
        })
        
        output$descr3 <- renderText({
            paste('This project applies the NLP techniques Word2Vec',
                  'and TF-IDF to wine review text then uses cosine',
                  'distance to compare wines. You can learn more at',
                  'the project page: github.com/pjh2011/wine_project')
        })
    }
)