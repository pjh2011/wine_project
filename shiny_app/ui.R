#ui.R

fluidPage(
    theme = "bootstrap.css",
    navbarPage("Wine2Vec Wine Searcher",
        tabPanel("Description", 
                 fluidRow(
                        h4(textOutput("descr1"), align = "center"),
                        h4(textOutput("descr2"), align = "center"),
                        h4(textOutput("descr3"), align = "center")
                     )
                 ),
        tabPanel("Search Wine Database",
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
                             h3(strong('+'), align = "center")
                         ),
                         
                         fluidRow(
                             selectInput('inPosWords', 
                                         label = "Positive Attributes", 
                                         choices = NULL, 
                                         selectize = TRUE,
                                         multiple = TRUE)
                         ),
                         
                         fluidRow(
                             h3(strong('-'), align = "center")
                         ),
                         
                         fluidRow(
                             selectInput('inNegWords', 
                                         label = "Negative Attributes", 
                                         choices = NULL, 
                                         selectize = TRUE,
                                         multiple = TRUE)
                         )
                     ),
                     
                     mainPanel(
                         uiOutput("similarWines")
                     )
                 ))
    )

)
