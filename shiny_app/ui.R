#ui.R

fluidPage(
    theme = "bootstrap.css",
    headerPanel("Wine2Vec Wine Searcher"),
    fluidRow(
        selectInput('inWineName', 
                    label = "Wine Name", 
                    choices = NULL, 
                    selectize = TRUE,
                    multiple = TRUE)
    ),
    
    fluidRow(
        selectInput('inPosWords', 
                    label = "Attributes to add to search", 
                    choices = NULL, 
                    selectize = TRUE,
                    multiple = TRUE)
    ),
    
    fluidRow(
        selectInput('inNegWords', 
                    label = "Attributes to subtract from search", 
                    choices = NULL, 
                    selectize = TRUE,
                    multiple = TRUE)
    ),
    fluidRow(
        textOutput("text1")
    )
)
