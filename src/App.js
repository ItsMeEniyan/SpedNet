import React from 'react';
import './App.css';
import MicRecorder from 'mic-recorder-to-mp3';

const Mp3Recorder = new MicRecorder({ bitRate: 128 });

class App extends React.Component {
  constructor(props){
    super(props);
    this.state = {
      isRecording: false,
      blobURL: '',
      isBlocked: false,
      apiResponse: '',
    };
  }

  callAPI() {
    fetch('http://localhost:9000/testAPI')
        .then(res => res.text())
        .then(res => this.setState({ apiResponse: res }));
}

  start = () => {
    if (this.state.isBlocked) {
      console.log('Permission Denied');
    } else {
      Mp3Recorder
        .start()
        .then(() => {
          this.setState({ isRecording: true });
        }).catch((e) => console.error(e));
    }
  };

  stop = () => {
    Mp3Recorder
      .stop()
      .getMp3()
      .then(([buffer, blob]) => {
        var fd = new FormData();
        fd.append('audio',blob);

        const blobURL = URL.createObjectURL(blob)
        this.setState({ blobURL, isRecording: false });
      }).catch((e) => console.log(e));
  };

  componentDidMount() {
    this.callAPI();
    navigator.getUserMedia({ audio: true },
      () => {
        console.log('Permission Granted');
        this.setState({ isBlocked: false });
      },
      () => {
        console.log('Permission Denied');
        this.setState({ isBlocked: true })
      },
    );
  }

  render(){
    return (
      <div className="App">
        <header className="App-header">
          <h1>{this.state.apiResponse}</h1>
          <div>
          <button className="button green" onClick={this.start} disabled={this.state.isRecording}>Start Recording</button>
          <button className="button red" onClick={this.stop} disabled={!this.state.isRecording}>Stop and Submit</button>
          </div>
          <audio src={this.state.blobURL} controls="controls" />
        </header>
      </div>
    );
  }
}

export default App;
