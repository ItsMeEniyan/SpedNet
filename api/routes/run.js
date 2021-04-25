const path = require('path');
const multer = require('multer');
//const Mp32Wav = require('mp3-to-wav');
//path.resolve('../uploads/audio-1619250145998.mp3')
//const nmp3wav = new Mp32Wav('D:\\Github\\SpedNet\\api\\uploads\\audio-1619250145998.mp3','D:\\Github\\SpedNet\\api\\uploads');
//nmp3wav.decodeMp3();
//nmp3wav.exec();
//console.log(path.resolve("./uploads/audio-1619250145998.mp3"));
//path.resolve("./uploads/wav")
const ffmpeg = require('fluent-ffmpeg');
//let track = './uploads/audio-1619250145998.mp3';//your path to source file
/*
let track = './uploads/'+req.file.filename;
ffmpeg(track)
.toFormat('wav')
.on('error', (err) => {
    console.log('An error occurred: ' + err.message);
})
.on('progress', (progress) => {
    // console.log(JSON.stringify(progress));
    console.log('Processing: ' + progress.targetSize + ' KB converted');
})
.on('end', () => {
    console.log('Processing finished !');
})
.save('./uploads/wav/audio.wav');//path where you want to save your file
*/
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join('./uploads'))
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + '.mp3')
        //console.log(req.file.filename);
    }
})

var upload = multer({ storage: storage })
//var upload = multer({ dest: 'uploads/' })


var express = require('express');
var router = express.Router();
//var app = express(); 

/*
router.get('/', function(req, res, next) {
    res.send('API is working properly');
});*/

/*app.listen(9000, function() { 
    console.log('server running on port 9000'); 
} ) */

// Function callName() is executed whenever  
// url is of the form localhost:3000/name 
router.post('/', upload.single('audio'), callName);



function callName(req, res) {
    console.log(req.file);
    let track = './uploads/' + req.file.filename;
    let tempwav = 'hi' + Date.now() + '.wav';
    console.log(track);
    console.log(tempwav);
    ffmpeg(track)
        .toFormat('wav')
        .on('error', (err) => {
            console.log('An error occurred: ' + err.message);
        })
        .on('progress', (progress) => {
            // console.log(JSON.stringify(progress));
            console.log('Processing: ' + progress.targetSize + ' KB converted');
        })
        .on('end', () => {
            console.log('Processing finished !');
        })
        .save('./uploads/wav/' + tempwav);//path where you want to save your file


    // Use child_process.spawn method from  
    // child_process module and assign it 
    // to variable spawn 
    var spawn = require("child_process").spawn;
    var exec = require("child_process").exec;

    // Parameters passed in spawn - 
    // 1. type_of_script 
    // 2. list containing Path of the script 
    //    and arguments for the script  

    // E.g : http://localhost:3000/name?firstname=Mike&lastname=Will 
    // so, first name = Mike and last name = Will 
    /*var process = spawn('python',["C:\\Users\\sreyus\\anaconda3\\Scripts\\conda run -n tensorflow python final.py",'D:\\Github\\SpedNet\\api\\uploads\\wav\\'+tempwav/*, 
                            req.query.firstname, 
req.query.lastname] ); */
    //var process = spawn('C:\\Users\\sreyus\\anaconda3\\Scripts\\conda',["run"," -n", "tensorflow", "python", "final.py",'D:\\Github\\SpedNet\\api\\uploads\\wav\\'+tempwav] );
    exec('C:\\Users\\sreyus\\anaconda3\\Scripts\\conda run -n tensorflow python model/final.py /wav/' +tempwav , function (err, stdout, stderr) {
        return res.send(stdout);
    });
    // Takes stdout data from script which executed 
    // with arguments and send this data to res object 
    // process.stdout.on('data', function(data) { 
    //     return res.send(data.toString()); 
    // } ) 

    // process.on('close', (code) => {
    //     if (code > 0){
    //     console.log(code);
    //       return res.status(500).json({ msg: "Internal server error" });
    // }});
}

module.exports = router;