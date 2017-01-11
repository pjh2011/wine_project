#ui.R

fluidPage(
    theme = "bootstrap.css",
    navbarPage("Wine2Vec Wine Searcher",
        tabPanel("Description", 'hello'),
        tabPanel("Search Wine DB",
                 sidebarLayout(
                     sidebarPanel(
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
                         )
                     ),
                     
                     mainPanel(
                         textOutput("text1")
                     )
                 ))
    )

)
