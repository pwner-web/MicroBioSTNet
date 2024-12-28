# Libraries
library(ggraph)
library(igraph)
library(tidyverse)
 
# create a data frame giving the hierarchical structure of your individuals
set.seed(1234)
d1 <- data.frame(from="origin", to=paste("group", seq(1,10), sep=""))
d2 <- data.frame(from=rep(d1$to, each=10), to=paste("subgroup", seq(1,100), sep="_"))
hierarchy <- rbind(d1, d2)
 
# create a dataframe with connection between leaves (individuals)
all_leaves <- paste("subgroup", seq(1,100), sep="_")
connect <- rbind( 
  data.frame( from=sample(all_leaves, 100, replace=T) , to=sample(all_leaves, 100, replace=T)), 
  data.frame( from=sample(head(all_leaves), 30, replace=T) , to=sample( tail(all_leaves), 30, replace=T)), 
  data.frame( from=sample(all_leaves[25:30], 30, replace=T) , to=sample( all_leaves[55:60], 30, replace=T)), 
  data.frame( from=sample(all_leaves[75:80], 30, replace=T) , to=sample( all_leaves[55:60], 30, replace=T)) )
connect$value <- runif(nrow(connect))

connect <- read.csv("checkpoints/network_edge.csv", encoding="UTF-8")
head(connect)
hierarchy <- read.csv("checkpoints/network_node.csv", encoding="UTF-8")
head(hierarchy)

#print("connect")
#head(connect)
#print("hierarchy")
#head(hierarchy)
#print(length(unique(c(as.character(hierarchy$from), as.character(hierarchy$to)))))
# create a vertices data.frame. One line per object of our hierarchy
vertices  <-  data.frame(
  name = unique(c(as.character(hierarchy$from), as.character(hierarchy$to))) , 
  value = runif(length(unique(c(as.character(hierarchy$from), as.character(hierarchy$to)))))
) 
# Let's add a column with the group of each name. It will be useful later to color points
vertices$group  <-  hierarchy$from[ match( vertices$name, hierarchy$to ) ]

# Create a graph object
mygraph <- graph_from_data_frame( hierarchy, vertices=vertices )
 
# The connection object must refer to the ids of the leaves:
from  <-  match( connect$from, vertices$name)
to  <-  match( connect$to, vertices$name)

#print("vertices")
#print(vertices)
#print("mygraph")
#print(mygraph)
#print("from")
#print(from)
#print("to")
#print(to)
# Basic usual argument
p=ggraph(mygraph, layout = 'dendrogram', circular = TRUE) + 
  geom_conn_bundle(data = get_con(from = from, to = to), width=1, alpha=0.2, aes(colour=..index..)) +
  scale_edge_colour_distiller(palette = "RdPu") 
  #theme_void() +
  #theme(legend.position = "none")

## It is good to color the points following their group appartenance
library(RColorBrewer)
 
# And you can adjust the size to whatever variable quite easily!
p <- p + geom_node_point(aes(filter = leaf, x = x*1.05, y=y*1.05, colour=group, size=value, alpha=0.2)) +
  scale_colour_manual(values= rep( brewer.pal(9,"Paired") , 30)) +
  scale_size_continuous( range = c(0.1,10) ) +
  theme_void()  + 
  guides(colour = guide_legend(override.aes = list(alpha = 1), title="Phylum"),
         size = "none",
         alpha = "none",
         edge_colour = "none")
  #theme(
  #  #legend.position="none",
  #  plot.margin=unit(c(0,0,0,0),"cm"),
  #) +
  #expand_limits(x = c(-1.3, 1.3), y = c(-1.3, 1.3))

# 保存为jpg
ggsave("results/paper_edge_plot.jpg", plot = p, width = 10.4, height = 9, dpi = 300)
# 保存为png
ggsave("results/paper_edge_plot.png", plot = p, width = 10.4, height = 9, dpi = 300)
# 保存为pdf
ggsave("results/paper_edge_plot.pdf", plot = p, width = 10.4, height = 9)