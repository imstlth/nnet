extern crate ndarray;
use ndarray::prelude::*;
use plotters::prelude::*;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use std::time;
use std::env;
use image::GenericImageView;

const MIN_DEFAULT_WEIGHT: f64 = -2.0;
const MAX_DEFAULT_WEIGHT: f64 = 2.0;
const MIN_DEFAULT_BIAIS: f64 = -2.0;
const MAX_DEFAULT_BIAIS: f64 = 2.0;

const HELP_MSG: &str = "Usage: nnet [OPTION]... DOSSIER CONFIG
nnet permet d'entrainer un réseau de neurones sur une base de données d'entrainement du type MNIST.
Exemple: nnet /home/user/code/mnist/ 784,183,42,10
OPTION est l'ensemble des caractéristiques du réseau de neurones. Si elle ne sont pas renseignées, les caractéristiques par défaut seront utilisées.
  -H, --hidden-activation  La fonction d'activation des hidden layer parmi: Sigmoid ou ReLU
                             Par défaut: Sigmoid
  -O, --output-activation  La fonction d'activation du output layer parmi: Sigmoid ou Softmax
                             Par défaut: Sigmoid
  -s, --step               Le pas pour le gradient descent:
                             un grande valeur implique un apprentissage rapide mais peu précis et potentiellement qui ne fonctionne pas du tout.
                             une petite valeur implique un apprentissage lent mais précis et qui trouve potentiellement un minimum local.
                             Par défaut: 0.8
  -e, --epochs             Le nombre de fois que le réseau de neurone va s'entrainer. L'apprentissage est en général très important dans les premières époques mais plus lent par la suite, en fonction du step (-s).
                             Par défaut: 100
  -S, --mini-batch-size    La taille du mini batch pour la déscente de gradient. Plus la valeur est petite, plus l'apprentissage sera efficace mais lent. Et inversement.
                             Par défaut: 200
  -g, --graphic            Demande au programme de réalisé un graphique de son efficacité au cours des époques. Cela implique un temps de calcul plus long que juste l'entrainement du réseau de neurones.
                             Par défaut: désactivé
  -w, --width              La largeur en pixel du graphique réalisé pendant l'apprentissage du réseau. Ne fait rien si -g n'est pas spécifié.
                             Par défaut: 1000
  -h, --height             La hauteur en pixel du graphique réalisé pendant l'apprentissage du réseau. Ne fait rien si -g n'est pas spécifié.
                             Par défaut: 1000
  -l, --labels             Le nom du fichier sous format MNIST qui contient les labels pour l'entrainement du réseau.
                             Par défaut: train-labels.idx1-ubyte
  -i, --images             Le nom du fichier sous format MNIST qui contient les images pour l'entrainement du réseau.
                             Par défaut: train-images.idx3-ubyte

Divers:
  -v, --verbose            Permet d'afficher plus de détails dans la sortie du programme.
                             Par défaut: désactivé
      --help               Affiche ce message d'aide.
  -E, --encode-after       Après l'entrainement, demande au proramme d'enregistrer les biais et les poids dans le fichier « encoded.nnet » dans le chemin DOSSIER.
                             Par défaut: désactivé
  -D, --decode-instead     Au lieu d'entrainer le réseau avec la descente de gradient et mini batch etc..., on utilise les fonctions d'activations, les biais et les poids fournis par « encoded.nnet » dans le DOSSIER. ATTENTION : Il est nécéssaire que le réseau encodé ait la même CONFIG que le réseau actuel !
                             Par défaut: désactivé

DOSSIER est le dossier contenant les fichiers d'entrainement pour le réseau de neurones.
Le nom de ces fichiers par défaut est : « train-labels.idx1-ubyte » et « train-images.idx3-ubyte ».
Il est possible de modifier ça avec les options -l et -i.

CONFIG est une suite de nombre indiquant l'organisation du réseau de neurones.
Le premier nombre indique uniquement le nombre d'entrées et ne correspond pas à une couche du réseau.
Exemple: « 784,88,10 » signifie que le réseau doit prendre 784 entrées, qu'il possède 1 seule hidden layer avec 88 neurones et une couche de sortie de 10 neurones.
La liste des nombres doit être séparée par des virgules et ne doit pas contenir d'espace.

CODES D'ERREUR:
  0   Le programme a fonctionné correctement.
  2   Mauvaise utilisation de la commande nnet.
  74  Erreur liée à l'ouverture des fichiers.";

// On peut pas juste faire Array<f64, Ix2> car une matrix est obligatoirement un rectangle or ce
// n'est pas le cas d'un réseau de neurone
// Autrement dit, dans une matrix y fait toujours la même taille, alors que les couches ne font pas
// la même taille
type Layer = Array<f64, Ix1>;
type Network = Vec<Layer>;

// On utilise un seul struct qui contient toutes les données du réseau
struct Net<'train> {
    // Les poids pour 1 couche peuvent être représentés sous la forme d'un Array 2D avec :
    // x = les neurones dont le poids est celui de leur sortie
    // y = les neurones dont le poids est celui de l'entrée
    // Ici, on a les poids pour chaque couche
    weights: Vec<Array<f64, Ix2>>,
    // Les biais, les activations et les sums suivent le type Network car il y en a
    // seulement 1 à chaque neurone
    biais: Network,
    activations: Network,
    sums: Network,
    // Le training est un Array 2D avec :
    // x = les différentes ensemble d'entrées/sorties possible
    // y = les signaux d'entrées/sorties
    training_input: &'train Array<f64, Ix2>,
    training_output: &'train Array<f64, Ix2>,
    activation_fns: [ActivationFn; 2]
}

// Les fonctions d'activation
#[derive(Clone, Copy)]
enum ActivationFn {
    Sigmoid,
    ReLU,
    Softmax
}

// Chaque fonction d'activation s'applique directement sur l'entièreté de la couche
fn sigmoid(sums: &Layer) -> Layer {
    sums.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn relu(sums: &Layer) -> Layer {
    sums.mapv(|v| 0_f64.max(v))
}

fn softmax(sums: &Layer) -> Layer {
    let denominateur = sums.mapv(f64::exp).sum();
    sums.mapv(|v| v.exp() / denominateur)
}

// La dérivée des fonctions d'activations - excepté softmax évidemment
// - car softmax est seulement utilisé à la fois oublie pas
// Si on change la fonction d'activation, il faut choisir la bonne dérivation aussi
// Ici, on ne modifie pas directement la couche pour des raisons évidentes
fn sigmoid_derivative(sums: &Layer) -> Array<f64, Ix1> {
    let layer_sigmoid = sums.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    &layer_sigmoid * (1.0 - &layer_sigmoid)
}

fn relu_derivative(sums: &Layer) -> Array<f64, Ix1> {
    sums.mapv(|v| usize::from(v > 0.0) as f64)
}

impl<'train> Net<'train> {

    // Ce qui change par rapport à la version précédente est que le nombre d'entrées doit être
    // indiqué en tout premier dans config
    // En état de marche
    fn new(config: &Vec<usize>, activation_fns: [ActivationFn; 2], training_input: &'train Array<f64, Ix2>, training_output: &'train Array<f64, Ix2>) -> Net<'train> {

        // Génération des poids aléatoires entre le min et le max de défaut
        // Ainsi que la génération des biais
        // En état de marche
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biais = Vec::new();
        let mut activations = vec![Array::<f64, Ix1>::zeros(config[0])];
        let mut sums = Vec::new();
        // Pour chaque couche (il y en a config.len()-1 car le 1ere et le nombre d'entrées)
        for n_couche in 0..config.len()-1 {
            // On génère un Array 2D pour les poids
            weights.push(Array::from_shape_simple_fn(
                // Avec la forme:
                // y = les neurones dont le poids est celui de l'entrée - config[n_couche+1]
                // x = les neurones dont le poids est celui de leur sortie - config[n_couche]
                // Note: pour la première couche, x est le nombre d'entrées
                (config[n_couche+1], config[n_couche]),
                // On met des valeurs random entre le min et le max de défaut
                || rng.gen_range(MIN_DEFAULT_WEIGHT..=MAX_DEFAULT_WEIGHT)
            ));

            // On génère l'Array 1D pour les biais
            biais.push(Array::from_shape_simple_fn(
                config[n_couche+1],
                || rng.gen_range(MIN_DEFAULT_BIAIS..=MAX_DEFAULT_BIAIS)
            ));

            // On génère des Array 1D pour sums et activations remplis de 0
            activations.push(Array::<f64, Ix1>::zeros(config[n_couche+1]));
            sums.push(Array::<f64, Ix1>::zeros(config[n_couche+1]));
        }

        // Je sais pas s'il y a un moyen de réduire ça, parce que c'est pas beau
        Net {
            weights,
            biais,
            activations,
            sums,
            training_input,
            training_output,
            activation_fns
        }

        // Si t'as un doute sur les axes dans les infos du Net
        /* println!("{weights:#?}");
        println!();
        println!("{biais:#?}");
        println!();
        println!("{sums:#?}");
        println!();
        println!("{activations:#?}"); */
    }

    // On calcule la sortie du réseau et l'ensemble des activations et des sums
    // En état de marche
    fn run(&mut self, inputs: ArrayView<f64, Ix1>) -> Layer {
        // On remplie la première ligne des activations
        self.activations[0] = inputs.to_owned();

        for n_layer in 0..self.biais.len() {
            // On calcule en premier les sommes de la couche
            // C'est putain de incroyable de pouvoir faire ça juste comme ça. Une jouissance
            // À noter que activations[n_layer] et en vérité les activations du layer précédent car
            // activations est décalé de 1 avec les inputs
            self.sums[n_layer] = self.weights[n_layer].dot(&self.activations[n_layer]) + &self.biais[n_layer];
            // Maintenant calcul des activations
            // Pour savoir si on a besoin de la fonction d'activation des hidden layers ou du
            // output layer
            let fn_type = usize::from(n_layer == self.biais.len() - 1);
            // Cette fois-ci c'est bien n_layer+1 ce qui revient aux activations du bon layer
            self.activations[n_layer+1] = match self.activation_fns[fn_type] {
                    ActivationFn::ReLU => relu(&self.sums[n_layer]),
                    ActivationFn::Sigmoid => sigmoid(&self.sums[n_layer]),
                    ActivationFn::Softmax => softmax(&self.sums[n_layer]),
            }
        }
        // On renvoie les dernières activations (sorties)
        match self.activations.last() {
            Some(output) => output.clone(),
            None => {
                println!("Le réseau de neurone ne contient aucune couche");
                std::process::exit(2);
            }
        }
    }

    // Calcul du cout du réseau pour un certain pourcentage de training_input
    // En état de marche
    fn cost(&mut self, percent: f64) -> f64 {

        let mut score = 0.;

        // On génère une liste de longueur percent * len contenant les index des data du dataset
        // NOTE WARNING IMPORTANT = .len() existe pour les arrays mais il c'est le nombre total
        // d'éléments dans l'ensemble des dimensions. Pour avoir seulement la longueur d'une
        // dimension, il y a la méthode .dim()
        let training_percent = (self.training_input.dim().0 as f64 * percent).round() as usize;
        let mut rng = thread_rng();
        let mut indexes: Vec<usize> = (0..self.training_input.dim().0).collect();
        indexes.shuffle(&mut rng);

        // Pour chacun de ces index, on calcul la formule du cout
        for n_data in &indexes[0..training_percent] {
            // On récupère le couple input - output associés à n_data
            let input = self.training_input.slice(s![*n_data, ..]);
            let output = self.training_output.slice(s![*n_data, ..]);
            // On calcule la différence de output et de la sortie du réseau pour input
            let mut data_score = self.run(input) - output;
            // On la met au carré et on additionne le tout
            data_score.mapv_inplace(|d| d.powi(2));
            score += data_score.sum();
        }
        // On applique le reste de la formule
        score / (2. * percent * self.training_input.dim().0 as f64)
    }

    // La dérivée du cout - je sais pas si c'est vraiment ça avec le mini batch mais pas grave
    fn cost_derivative(network_output: &Layer, expected_output: &Layer) -> Array<f64, Ix1> {
        network_output - expected_output
    }

    // Calcul l'ensemble des gradients pour l'ensemble d'entrées et de sorties attendues
    fn gradient(&mut self, input: ArrayView<f64, Ix1>, output: ArrayView<f64, Ix1>) -> (Vec<Array<f64, Ix1>>, Vec<Array<f64, Ix2>>) {

        let mut nabla_weights = Vec::new();
        let mut nabla_biais = Vec::new();

        // On lance le réseau pour avoir les activations et les sums remplies
        let network_output = self.run(input).to_owned();
        let expected_output = output.to_owned();

        // BP1
        // On calcule l'error (le delta) pour la couche de sortie
        let mut delta = Net::cost_derivative(&network_output, &expected_output);

        // Alors pour la suite, j'ai **vraiment** essayé de trouver un moyen clean de le faire
        // mais ça demande de faire un produit extérieur sur des Array 3D alors que c'est même
        // pas implémenté pour des Vec
        // Donc, on va le faire à la zeub
        for n_layer in (0..self.biais.len()).rev() {
            // n_layer est la couche à calculer

            // On calcule le delta si c'est un hidden layer sinon il est déjà calculé
            if n_layer != self.biais.len() - 1 {

                // BP2
                // Putain c'est vraiment très beau de faire juste ça comme ça
                let sum_derivative = match self.activation_fns[0] {
                    ActivationFn::Sigmoid => sigmoid_derivative(&self.sums[n_layer]),
                    ActivationFn::ReLU => relu_derivative(&self.sums[n_layer]),
                    ActivationFn::Softmax => {
                        println!("Softmax ne devrait pas être présent dans un hidden layer !");
                        std::process::exit(2);
                    },
                };
                delta = self.weights[n_layer+1].clone().reversed_axes().dot(&delta) * sum_derivative;
            }

            // BP3
            nabla_biais.insert(0, delta.clone());

            // BP4
            // En vérité le BP4 est le produit extérieur (outer product) entre le delta et les
            // activations
            let weight_chgs = Net::outer(&delta, &self.activations[n_layer]);
            nabla_weights.insert(0, weight_chgs);
        }

        (nabla_biais, nabla_weights)
    }

    // Le produit extérieur
    fn outer(x: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> Array<f64, Ix2> {
        let (size_x, size_y) = (x.shape()[0], y.shape()[0]);
        let x_reshaped = x.view().into_shape((size_x, 1)).unwrap();
        let y_reshaped = y.view().into_shape((1, size_y)).unwrap();
        x_reshaped.dot(&y_reshaped)
    }

    // Pour mettre à jour le réseau en fonction de nabla_biais et nabla_weights
    fn update(&mut self, nabla_biais: Vec<Array<f64, Ix1>>, nabla_weights: Vec<Array<f64, Ix2>>, coeff: f64) {
        // C'est tellement  plus beau que l'ancienne version
        for n_couche in 0..self.biais.len() {
            self.biais[n_couche] -= &(&nabla_biais[n_couche] * coeff);
            self.weights[n_couche] -= &(&nabla_weights[n_couche] * coeff);
        }
    }

    // La déscente de gradient avec mini batch
    fn learn(&mut self, step: f64, mini_batch_size: usize, epochs: usize, printing: bool) {
        let coeff = step / mini_batch_size as f64;

        // On crée des Vec qui ont exactement la même forme que self.biais et self.weights mais
        // remplis de 0 pour pourvoir après additionner tous les gradients. On calcule ça ici pour
        // ne pas le calculer dans la boucle puisque ça demande quand même un peu de puissance
        let mut sum_nabla_biais_default = Vec::new();
        let mut sum_nabla_weights_default = Vec::new();
        for n_couche in 0..self.biais.len() {
            // On recré un Array qui a la même forme que biais/weights pour la couche mais avec des 0
            sum_nabla_biais_default.push(Array::<f64, Ix1>::zeros(self.biais[n_couche].raw_dim()));
            sum_nabla_weights_default.push(Array::<f64, Ix2>::zeros(self.weights[n_couche].raw_dim()));
        }

        for epoch in 0..epochs {
            if printing {
                println!("Epoch = {}/{epochs}", epoch+1);
            }

            // Les chunks dans training_input et training_output
            let mut training_input_chunks = self.training_input.axis_chunks_iter(Axis(0), mini_batch_size);
            let mut training_output_chunks = self.training_output.axis_chunks_iter(Axis(0), mini_batch_size);
            // Le nombre total de mini_batch
            let n_mini_batch = (self.training_input.dim().0 as f64 / mini_batch_size as f64).ceil() as usize;

            for _n_batch in 0..n_mini_batch {
                // Le batch en question. Un Array 2D : les entrées par la taille du batch
                // NOTE : si c'est le dernier batch, il est possible qu'il ne fasse la taille du
                // batch mais qu'il soit un peu plus petit
                let input_batch = training_input_chunks.next().unwrap();
                let output_batch = training_output_chunks.next().unwrap();
                // Les variables qui vont contenir les sommes ds nabla
                let mut sum_nabla_biais = sum_nabla_biais_default.clone();
                let mut sum_nabla_weights = sum_nabla_weights_default.clone();

                // Pour chaque couple (input, output) dans le batch
                for index in 0..input_batch.dim().0 {
                    // On calcule le gradient
                    let (nabla_biais, nabla_weights) = self.gradient(
                        input_batch.slice(s![index, ..]),
                        output_batch.slice(s![index, ..]));
                    // On les additionne aux sum
                    for n_couche in 0..nabla_biais.len() {
                        sum_nabla_biais[n_couche] += &nabla_biais[n_couche];
                        sum_nabla_weights[n_couche] += &nabla_weights[n_couche];
                    }
                }
                self.update(sum_nabla_biais, sum_nabla_weights, coeff);
            }
        }
    }

    // Tentative de créer un graph du cout en fonction de temps pour démontrer le gradient descent
    fn graph(&mut self, step: f64, mini_batch_size: usize, epochs: usize, width: u32, height: u32, filename: &str, verbose: bool) {
        // let initial_cost = self.cost(percent).ceil();
        let drawing_area = BitMapBackend::new(filename, (width, height))
            .into_drawing_area();

        drawing_area.fill(&WHITE).unwrap();

        let mut my_chart = ChartBuilder::on(&drawing_area)
            .margin(15)
            .caption("Evolution du nombre de bonnes réponses", ("Arial", 24))
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0..epochs, 0.0..100.0)
            .unwrap();

        my_chart.configure_mesh().draw().unwrap();

        // let mut couts = Vec::new();
        let mut guess = Vec::new();
        for epoch in 0..epochs {
            if verbose { println!("Epoch = {}/{epochs}", epoch+1); }
            // couts.push(self.cost(percent) / initial_cost);
            guess.push(self.note() as f64 / 600.);
            self.learn(step, mini_batch_size, 1, false);
        }

        // my_chart.draw_series(
            // LineSeries::new((0..).zip(couts), &BLACK)
        // ).unwrap();
        my_chart.draw_series(
            LineSeries::new((0..).zip(guess), &BLACK)
        ).unwrap();
    }

    // Cette fonction calcul la note du réseau de neurone (c'est-à-dire le nombre d'images qu'il a
    // correctement trouvé par rapport au nombre total d'image)
    fn note(&mut self) -> i32 {
        let mut count = 0;
        for n_data in 0..60_000 {
            let network_output = self.run(self.training_input.slice(s![n_data, ..]))
                .mapv(f64::round);
            let expected_output = self.training_output.slice(s![n_data, ..]).to_owned();
            if network_output == expected_output {
                count += 1;
            }
        }
        count
    }

}

// Decoder une base de données images MNIST en un array 2D (les images sont flatten)
fn mnist_decode_images(filename: &str) -> Array<f64, Ix2> {
    let path = Path::new(filename);
    let file = match File::open(path) {
        Ok(file_object) => file_object,
        Err(error) => {
            println!("Erreur lors de l'ouverture du fichier :\n{error}");
            std::process::exit(74);
        }
    };
    let mut bytes = file.bytes();
    // On ne vérifie pas les 16 premiers bytes
    for _i in 0..16 { bytes.next(); }
    // C'est incroyablement plus clean que l'ancienne version
    Array::<f64, Ix2>::from_shape_simple_fn((60_000, 784), || bytes.next().unwrap().unwrap() as f64 / 255.)
}

// Idem pour les labels MNIST
fn mnist_decode_labels(filename: &str) -> Array<f64, Ix2> {
    let path = Path::new(filename);
    let file = match File::open(path) {
        Ok(file_object) => file_object,
        Err(error) => {
            println!("Erreur lors de l'ouverture du fichier :\n{error}");
            std::process::exit(74);
        }
    };
    let mut bytes = file.bytes();
    // On ne vérifie pas les 8 premiers bytes
    for _i in 0..8 { bytes.next(); }
    let mut labels = Array::<f64, Ix2>::zeros((0, 10));
    for _n_label in 0..60_000 {
        // On doit transformer le numéro contenu dans le fichier en un Array avec 1 en sa position
        let mut label_input = array![0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
        let label = bytes.next().unwrap().unwrap();
        label_input[label as usize] = 1.;
        // On le rajoute à l'Array de départ
        labels.push(Axis(0), label_input.view()).unwrap();
    }
    labels
}

// Une fonction d'aide pour encode_weights_biais
fn push_w_space(content: &mut String, to_add: impl ToString) {
    content.push_str(&(to_add.to_string() + " "));
}

// Une fonction qui permet de sauvegarder les fonctions d'activation, les poids et les biais dans
// un fichier
// Le format prévu est le suivant :
// Chaque nombre est séparé des autres par un espace
// Le premier nombre est la fonction d'activation des hidden layers
// Le deuxième nombre est la fonction d'activation du output layer
//  0 = Sigmoid   1 = ReLU   2 = Softmax
// Le troisième nombre C est le nombre de couches dans le réseau de neurones
// Les C nombres suivants sont le nombre de neurones par couche
// Ensuite, on a tous les biais neurone par neurone, couche par couche
// Et puis, il y a tous les poids : neurones d'entrée par neurones d'entrées, pour chaque neurone
// récepteur du signal pondéré, pour chaque couche
fn encode_weights_biais(dossier: &str, network: &Net) {

    let mut file_content = String::new();
    // On rajoute les fonctions d'activation
    for activ_fn in &network.activation_fns {
        push_w_space(&mut file_content,
            match activ_fn {
                ActivationFn::Sigmoid => "0",
                ActivationFn::ReLU => "1",
                ActivationFn::Softmax => "2"
            });
    }
    // On rajoute le nombre de couche (en comptant les entrées)
    push_w_space(&mut file_content, network.activations.len());
    // Pour chaque couche et les entrées, on rajoute le nombre de neurones/d'entrées
    for couche in &network.activations {
        push_w_space(&mut file_content, couche.dim());
    }

    // On rajoute les biais dans l'ordre: pour chaque couche, neurone par neurone
    network.biais.iter()
        .map(
            |couche| couche.for_each(
                |biais| push_w_space(&mut file_content, biais)
            )
        )
        // On consume simplement le map
        .for_each(drop);

    // Idem pour les poids
    // On rajoute les poids dans l'ordre : pour chaque couche, pour chaque neurone récepteur,
    // neurones par neurones dont la sortie est pondérée par le poids en question
    network.weights.iter()
        .map(
            |arr| arr.for_each(
                // for_each visite les éléments comme prévu : pour chaque ligne, col par col
                // Ici, ça donne : pour chaque récepteur, émetteur par émetteur
                // (Je sais pas si les terms de "récepteur" et d'"émetteur" sont clairs)
                |weight| push_w_space(&mut file_content, weight)
            )
        )
        .for_each(drop);

    // On supprime le dernier espace
    file_content.pop();

    // On crée/ouvre le fichier seulement lors qu'on a finit de créer son futur contenu
    let fichier = dossier.to_string() + "encoded.nnet";
    let path = Path::new(&fichier);
    let mut file = match File::create(path) {
        Ok(file_object) => file_object,
        Err(error) => {
            println!("Erreur lors de la création/ouverture du fichier :\n{error}");
            std::process::exit(74);
        }
    };
    // On écrit les données
    match file.write_all(file_content.as_bytes()) {
        Ok(_) => (),
        Err(error) => {
            println!("Erreur lors de l'écriture du fichier :\n{error}");
            std::process::exit(74);
        }
    }
}

// L'inverse de la fonction juste au dessus
// Décoder le fichier contenant les fonctions d'activations, les biais et les poids
// pour remplir le réseau donné
fn decode_weights_biais(dossier: &str, network: &mut Net) {
    let fichier = dossier.to_string() + "encoded.nnet";
    let path = Path::new(&fichier);
    let mut file = match File::open(path) {
        Ok(file_object) => file_object,
        Err(error) => {
            println!("Erreur lors de l'ouverture du fichier :\n{error}");
            std::process::exit(74);
        }
    };

    let mut file_content = String::new();
    match file.read_to_string(&mut file_content) {
        Ok(_) => (),
        Err(error) => {
            println!("Erreur lors de la lecture du fichier :\n{error}");
            std::process::exit(74);
        }
    };
    let mut infos = file_content.split_whitespace();
    // On change les fonctions d'activations
    let correspondance = [ActivationFn::Sigmoid, ActivationFn::ReLU, ActivationFn::Softmax];
    for activ_type in 0..2 {
        let activ_fn = infos.next().unwrap().parse::<usize>().unwrap();
        network.activation_fns[activ_type] = correspondance[activ_fn];
    }
    // On vérifie qu'il y a le bon nombre de couches
    if infos.next().unwrap().parse::<usize>().unwrap() != network.activations.len() {
        println!("La CONFIG ne correspond pas.");
        std::process::exit(2);
    }
    // Pour chaque couche, on vérifie que ça correspond bien (+ les entrées)
    for n_layer in 0..network.activations.len() {
        if infos.next().unwrap().parse::<usize>().unwrap() != network.activations[n_layer].dim() {
            println!("La CONFIG ne correspond pas.");
            std::process::exit(2);
        }
    }

    // Maintenant, on sait que la CONFIG correspond bien donc on peut tout remplir
    // Pour chaque couche, on transfère les biais de la couche
    for biais_layer in &mut network.biais {
        // On prend le nombre de biais souhaité
        // Si on ne fait pas .by_ref(), ça move infos
        let file_biais = infos.by_ref()
            .take(biais_layer.dim());
        // On change la config du réseau actuel
        *biais_layer = file_biais.map(
            // Pour avoir le bon type
            |n| n.parse().unwrap()
        ).collect();
    }

    // Maintenant pour les poids
    for weights_layer in &mut network.weights {
        // On multiplie les 2 dimensions. C'est assez mal fait de la part de ndarray par contre
        let dim_product = weights_layer.raw_dim()
            .as_array_view_mut()
            .product();
        // Encore une fois, si on ne fait pas by_ref(), ça move infos
        let file_weights_flat = infos.by_ref()
            .take(dim_product)
            .map(|n| n.parse().unwrap());
        // On transforme les poids qui sont flat en un Array 2D avec les bonnes dimensions
        // C'est très moche mais c'est le seul moyen de s'assurer que l'array produit soit en 2D
        let weights_layer_shape = (weights_layer.shape()[0], weights_layer.shape()[1]);
        *weights_layer = Array::from_iter(file_weights_flat)
            // On le moule comme dans un moule à gâteau :)
            .into_shape(weights_layer_shape)
            .unwrap();
    }
}

// L'interface interactive pour l'utilisateur
fn interactive(network: &mut Net) {
    loop {
        // On attend que l'utilisateur entre une information
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        let image_input;
        // On quitte proprement
        if input == "q" {
            std::process::exit(0);
        // Si on rentre un nombre, on fait avec le nombre de la base de données
        } else if input.chars().all(char::is_numeric) {
            // L'image en question
            image_input = network.training_input.slice(
                s![input.parse::<usize>().unwrap(), ..]
            ).to_owned();
        // Sinon, c'est que l'entrée est un chemin et on ouvre l'image
        } else {
            // On tente d'ouvrir l'image
            let img = match image::open(input) {
                Ok(image_object) => image_object,
                Err(error) => {
                    println!("Erreur lors de l'ouverture de l'image : {error}");
                    continue
                }
            };

            // On remarque que les format utilisé pour les coulerus par le MNIST est :
            // 00 = white  et ff = black
            // alors que c'est le contraire dans les conventions. Il faut donc inverser le pixel
            image_input = Array::from_iter(
                img.pixels()
                .map(|p| 1. - (p.2[0] as f64 / 255.))
            );
        }
        println!();
        println!("Image contenue dans le fichier :");
        for row in 0..28 {
            for pixel in 0..28 {
                if image_input[(row * 28 + pixel) as usize] < 0.5 {
                    print!(" ");
                } else {
                    print!("%");
                }
            }
            println!();
        }
        println!();
        // On construit un HashMap à partir de l'Array 1D
        let mut number = -1;
        let mut results: Vec<(i32, f64)> = network.run(image_input.view())
            .iter()
            .map(|v| {
                number += 1;
                (number, *v)
            })
            .collect();
        // On sort le vec comme ça la sortie est plus claire
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // On affiche les résultats
        for result in results {
            // On fait un pourcentage avec 2 décimal de précision
            println!("{}: {}", result.0, (result.1 * 10000.).round() / 100.);
        }
    }
}

fn mini_help() {
    println!("Usage: nnet [OPTION]... DOSSIER CONFIG");
    println!("Exécutez « nnet --help » pour obtenir des renseignements supplémentaires.");
}

// Fonction d'aide pour parse_args
// Elle récupère la valeur qui suit un argument du type --something
// Si cette argument n'existe pas, ou qu'il n'y a pas de valeur qui le suit, elle renvoie une
// valeur par défaut qui est renseigné dans les paramètres de la fonction
fn get_arg<F>(predicate: F, args: &Vec<String>, default: String) -> String
where F: Fn(&String) -> bool {
    // On extrait l'emplacement du flag
    let arg_pos = args.iter()
        .position(predicate);
    // S'il existe bien sur
    if let Some(pos) = arg_pos {
        // Si une valeur le suit, on la renvoie, sinon on affiche la mini aide et on quitte
        match args.get(pos + 1) {
            Some(value) => return value.clone(),
            None => {
                mini_help();
                std::process::exit(2);
            }
        }
    // S'il n'existe pas, on renvoie la valeur par défaut
    } else {
        default
    }
}

// Renvoie le contenu des arguments passés à la ligne de commande
// On a dans l'ordre :
// 1) Les fonctions d'activation
// 2) Le step
// 3) Le nombre d'époques
// 4) La taille du mini batch
// 5) S'il faut faire un graphique
// 6) La largeur du graphique
// 7) La hauteur du graphique
// 8) Le nom du fichier qui contient les labels
// 9) Le nom du fichier qui contient les images
// 10) S'il faut être verbose
// 11) S'il faut encoder le réseau une fois qu'il est entrainé
// 12) S'il faut utiliser des données fournies au lieu de l'entrainer
// 13) Le chemin vers ces fichiers et là où le graphique serait généré
// 14) La config du réseau de neurones (le nombre d'entrées et le nombre de neurones par couche)
fn parse_args() -> ([ActivationFn; 2], f64, usize, usize, bool, u32, u32, String, String, bool, bool, bool, String, Vec<usize>) {
    let mut args: Vec<String> = env::args().collect();
    // Si on demande explicitement le message d'aide, on l'affiche
    if args.contains(&String::from("--help")) {
        println!("{}", HELP_MSG);
        std::process::exit(0);
    // S'il n'y a pas assez d'arguments, on rappelle le mini message d'aide
    } else if args.len() < 3 {
        mini_help();
        std::process::exit(2);
    // Sinon, on extrait les informations des arguments de la ligne de commande
    } else {
        // On extrait les fonctions d'activation
        let mut activation_fns = [ActivationFn::Sigmoid, ActivationFn::Sigmoid];

        // Pour les hidden layers
        let hid_act_flag = |arg: &String| arg == "-H" || arg == "--hidden-activation";
        // La valeur de l'activation pour les hidden layers donnée dans la ligne de commande
        let hidden_activation = get_arg(hid_act_flag, &args, String::new());
        // On prend seulement en compte ReLU car Sigmoid est la valeur par défaut et Softmax ne
        // sert pas dans les hidden layers
        if hidden_activation == "ReLU" {
            activation_fns[0] = ActivationFn::ReLU;
        }

        // Pour le output layer
        let out_act_flag = |arg: &String| arg == "-O" || arg == "--output-activation";
        // La valeur de l'activation pour le output layer dans la ligne de commande
        let output_activation = get_arg(out_act_flag, &args, String::new());
        // Là, on prend seulement en compte Softmax car Sigmoid est la valeur par défaut est ReLU
        // ne sert pas dans le output layer
        if output_activation == "Softmax" {
            activation_fns[1] = ActivationFn::Softmax;
        }


        // La valeur du step
        let step_flag = |arg: &String| arg == "-s" || arg == "--step";
        let step_str = get_arg(step_flag, &args, String::from("0.8"));
        let step = match step_str.parse::<f64>() {
            Ok(step_value) => step_value,
            Err(error) => {
                println!("Erreur lors de la lecture du step:\n{error}\n");
                mini_help();
                std::process::exit(2);
            }
        };

        // Le nombre d'epochs
        let epochs_flag = |arg: &String| arg == "-e" || arg == "--epochs";
        let epochs_str = get_arg(epochs_flag, &args, String::from("100"));
        let epochs = match epochs_str.parse::<usize>() {
            Ok(epochs_value) => epochs_value,
            Err(error) => {
                println!("Erreur lors de la lecture du nombre d'époques:\n{error}\n");
                mini_help();
                std::process::exit(2);
            }
        };

        // La taille du mini_batch
        let b_size_flag = |arg: &String| arg == "-S" || arg == "--mini-batch-size";
        let b_size_str = get_arg(b_size_flag, &args, String::from("200"));
        let mini_batch_size = match b_size_str.parse::<usize>() {
            Ok(size) => size,
            Err(error) => {
                println!("Erreur lors de la lecture de la taille du mini batch:\n{error}\n");
                mini_help();
                std::process::exit(2);
            }
        };

        // Si on doit faire un graphic
        let graph = args.contains(&String::from("-g")) || args.contains(&String::from("--graphic"));

        // Les arguments qui lui sont reliés
        let width_flag = |arg: &String| arg == "-w" || arg == "--width";
        let width_str = get_arg(width_flag, &args, String::from("1000"));
        let width = match width_str.parse::<u32>() {
            Ok(width_px) => width_px,
            Err(error) => {
                println!("Erreur lors de la lecture de la largeur du graphique:\n{error}\n");
                mini_help();
                std::process::exit(2);
            }
        };
        let height_flag = |arg: &String| arg == "-h" || arg == "--height";
        let height_str = get_arg(height_flag, &args, String::from("1000"));
        let height = match height_str.parse::<u32>() {
            Ok(height_px) => height_px,
            Err(error) => {
                println!("Erreur lors de la lecture de la hauteur du graphique:\n{error}\n");
                mini_help();
                std::process::exit(2);
            }
        };

        // Les arguments pour les noms des fichiers
        let label_flag = |arg: &String| arg == "-l" || arg == "--labels";
        let labels = get_arg(label_flag, &args, String::from("train-labels.idx1-ubyte"));
        let images_flag = |arg: &String| arg == "-i" || arg == "--images";
        let images = get_arg(images_flag, &args, String::from("train-images.idx3-ubyte"));

        // verbose
        let verbose = args.contains(&String::from("-v")) || args.contains(&String::from("--verbose"));

        // S'il faut encode le réseau après l'entrainement
        let encode_after = args.contains(&String::from("-E")) || args.contains(&String::from("--encode-after"));
        // S'il faut utiliser les biais et les poids fournis par un fichier
        let decode_instead = args.contains(&String::from("-D")) || args.contains(&String::from("--decode-instead"));


        // Maintenant, on passe à DOSSIER et CONFIG
        args.reverse();
        let dossier = match args.get(1) {
            Some(path) => path.clone(),
            None => {
                mini_help();
                std::process::exit(2);
            }
        };

        // CONFIG
        let config: Vec<usize> = match args.get(0) {
            // OK, là je me suis fais plaisir
            // S'il y a une config dans les arguments
            Some(config_list) => {
                // On la clone
                config_list.clone()
                    // On la split aux virgules pour en faire un iter
                    .split(",")
                    // Pour chaque valeur, on va la parse
                    .map(
                        |n|
                        n.parse::<usize>()
                        // Si le parse renvoie une erreur, on stop tout et on affiche la mini aide
                        .unwrap_or_else( |error| {
                                println!("Erreur lors de la lecture de la config:\n{error}\n");
                                mini_help();
                                std::process::exit(2)
                            }
                        )
                    )
                    // On collect l'iterator en un Vec<usize>
                    .collect()
            },
            None => {
                mini_help();
                std::process::exit(2);
            }
        };

        // Et on renvoie tout
        (activation_fns, step, epochs, mini_batch_size, graph, width, height, labels, images, verbose, encode_after, decode_instead, dossier, config)
    }
}

fn main() {
    let (activation_fns, step, epochs, mini_batch_size, graph, width, height, labels, images, verbose, encode_after, decode_instead, dossier, config) = parse_args();
    println!("Extraction des labels du dataset...");
    let training_output = mnist_decode_labels(&(dossier.clone() + &labels));
    println!("Fait!\n");
    println!("Extraction des images du dataset...");
    let training_input = mnist_decode_images(&(dossier.clone() + &images));
    println!("Fait!\n");
    println!("Création du réseau de neurones...");
    let mut mnist = Net::new(&config, activation_fns, &training_input, &training_output);
    println!("Fait!\n");
    // Si on utilise des biais et des poids fournis, on les décode maintenant
    if decode_instead {
        println!("Décodage du réseau de neurone enregistré...");
        decode_weights_biais(&dossier, &mut mnist);
        println!("Fait!\n");
    // Sinon on les génère
    } else {
        println!("Entrainement du réseau de neurone...");
        let start = time::Instant::now();
        println!("Calcul du cout initial...");
        let cost = mnist.cost(1.);
        let diff = start.elapsed();
        println!("Fait!");
        println!("SCORE = {cost}");
        println!("Temps pris pour calculer le cout de 100% du dataset = {} secondes", diff.as_secs());
        let start = time::Instant::now();
        // Si il est demandé de générer un graphique, on appelle la fonction
        if graph {
            println!("Apprentissage du réseau de neurone et création du graphique...");
            println!("NOTE: Le graphic se trouve en {dossier}graphic.png\n");
            mnist.graph(step, mini_batch_size, epochs, width, height, &(dossier.clone() + "graphic.png"), verbose);
        // Sinon, on fait la fonction normale
        } else {
            println!("Apprentissage du réseau de neurone...\n");
            mnist.learn(step, mini_batch_size, epochs, verbose);
        }
        let diff = start.elapsed();
        println!("Fait!");
        println!("Temps pris pour l'entrainement du réseau = {} minutes.\n", (diff.as_secs_f64() / 6.).round() / 10.);
    }
    // Dans tout les cas, on calcul le score
    println!("Calcul du cout final...");
    let cost = mnist.cost(1.);
    println!("Fait!");
    println!("SCORE = {cost}");
    println!("Calcul de la note du réseau de neurones...");
    let correct_guess = mnist.note();
    println!("Fait!");
    println!("Sur 60 000 images, le réseau de neurone en a trouvé {correct_guess}");
    // S'il faut enregistrer le réseau, on le fait maintenant
    if encode_after {
        println!("Enregistrement du réseau de neurone...");
        encode_weights_biais(&dossier, &mnist);
        println!("Fait!\n")
    }

    // On lance l'interface interactive où l'on peut fournir des images au réseau de neurone et il
    // répond automatique quel est sa réponse
    println!("Lancement de l'interface interactive :");
    println!("Vous pouvez fournir le chemin d'un fichier image de 28 pixels par 28 pixels et le réseau de neurone renverra quel chiffre pense-t-il que l'image représente.");
    println!("Vous pouvez aussi rentrer un nombre entre 1 et 60 000 et le réseau de neurone utilisera l'image correspondante de la base de donnée.");
    println!("Pour quitter l'interface interactive, écrivez 'q'.");
    interactive(&mut mnist);
    println!("Au revoir !");
}
