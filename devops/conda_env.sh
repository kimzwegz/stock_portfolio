## script to create virtual environment with Anaconda

echo creating virtual environment ...

read -p  "select directory to create your env:"$'\n' dir
echo $dir selected 

echo $'\n'

read -p "type name of directory to be created":$'\n' envdir
echo $envdir selected

echo $'\n'

condaenv=${dir}/$envdir
echo $condaenv

read -p "select requirements file path"$'\n' req
echo requirements file path selected: $req

echo $'\n'

reqs=$(<$req)

echo dependencies are:
echo $reqs

conda create --prefix $condaenv $reqs

#source /Users/karimkhalil/Coding/TPQ/my_files/devops/activate.sh
#conda activate $condaenv



