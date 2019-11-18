import pandas as pd
import json

input_file = pd.read_csv('movies_metadata.csv')
num_rows = len(input_file.index)
outdf = pd.DataFrame(index=range(num_rows), columns=['id', 'title', 'genres', 'posterpath'])
outdf['id'] = input_file['id']
outdf['title'] = input_file['title']
outdf['genres'] = input_file['genres']
outdf['posterpath'] = input_file['poster_path']
outdf.set_index('id', inplace = False)

nullGenres = outdf[outdf['genres'] == '[]'].index
outdf.drop(nullGenres, inplace=True)
nullPosters = outdf[outdf['posterpath'].isnull()].index
outdf.drop(nullPosters, inplace=True)
dropIds = ['106605','121351','140470','156415','1997-08-20','2012-09-29','2014-01-01','215908','23022','31772','35810','38585','53571','55602','56325','77621','79968']
outdf = outdf[~outdf['id'].isin(dropIds)]

print("Sorting genres...")
#genres = {}
for index, row in outdf.iterrows():
	#print(row)
	row = row.copy()
	genres = json.loads(row['genres'].replace("'", "\""))
	for i in range(len(genres)):
		genre = genres[i]["name"]
		if(genre not in outdf.columns):
			outdf.insert(len(outdf.columns), genre, int(0))
		outdf.loc[index, genre] = 1

#print(outdf.head(1))

print("Exporting...")
export_csv = outdf.to_csv('movies_metadata_processed.csv', index = None, header = True)